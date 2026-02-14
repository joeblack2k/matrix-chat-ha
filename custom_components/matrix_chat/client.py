"""Matrix API client for Matrix Chat custom integration."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import shutil
import tempfile
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

from aiohttp import ClientError, ClientSession, FormData
from homeassistant.helpers.storage import Store
from PIL import Image

from .const import DOMAIN, FORMAT_HTML

_LOGGER = logging.getLogger(__name__)


class MatrixChatError(Exception):
    """Base error for Matrix Chat."""


class MatrixChatAuthError(MatrixChatError):
    """Authentication error for Matrix Chat."""


class MatrixChatConnectionError(MatrixChatError):
    """Connectivity error for Matrix Chat."""


def _normalize_token(token: str | None) -> str:
    if not token:
        return ""
    out = token.strip()
    if out.lower().startswith("bearer "):
        out = out[7:].strip()
    return out


def _media_msgtype(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return "m.image"
    if mime_type.startswith("video/"):
        return "m.video"
    return "m.file"


def _sniff_mime_type(path: Path) -> str | None:
    """Best-effort MIME sniffing from file signature.

    We prefer sniffing over filename extension because files in /config/www may be
    misnamed (e.g. .jpg containing a PNG), which can break Matrix clients.
    """
    try:
        with path.open("rb") as f:
            head = f.read(32)
    except OSError:
        return None

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "video/mp4"
    return None


def _image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


async def async_validate_credentials(
    session: ClientSession,
    homeserver: str,
    user_id: str,
    password: str,
    access_token: str,
    verify_ssl: bool,
) -> dict[str, str]:
    """Validate provided credentials and return normalized auth details."""
    homeserver = homeserver.rstrip("/")
    token = _normalize_token(access_token)
    ssl_arg = None if verify_ssl else False

    if token:
        whoami_url = f"{homeserver}/_matrix/client/v3/account/whoami"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            async with session.get(whoami_url, headers=headers, ssl=ssl_arg) as resp:
                text = await resp.text()
                if resp.status == 200:
                    data = json.loads(text)
                    return {
                        "user_id": data.get("user_id", user_id),
                        "access_token": token,
                        "device_id": data.get("device_id", ""),
                    }
                if resp.status in (401, 403) and not password:
                    raise MatrixChatAuthError("Access token rejected and no password provided")
        except ClientError as err:
            raise MatrixChatConnectionError(f"Cannot reach Matrix homeserver: {err}") from err

    if not password:
        raise MatrixChatAuthError("Provide a password or a valid access token")

    login_url = f"{homeserver}/_matrix/client/v3/login"
    payload = {
        "type": "m.login.password",
        "identifier": {"type": "m.id.user", "user": user_id},
        "user": user_id,
        "password": password,
        "initial_device_display_name": "Home Assistant Matrix Chat",
    }

    try:
        async with session.post(login_url, json=payload, ssl=ssl_arg) as resp:
            text = await resp.text()
            if resp.status == 200:
                data = json.loads(text)
                return {
                    "user_id": data.get("user_id", user_id),
                    "access_token": _normalize_token(data.get("access_token", "")),
                    "device_id": data.get("device_id", ""),
                }
            if resp.status in (401, 403):
                raise MatrixChatAuthError("Invalid Matrix credentials")
            raise MatrixChatConnectionError(
                f"Matrix login failed with HTTP {resp.status}: {text[:400]}"
            )
    except ClientError as err:
        raise MatrixChatConnectionError(f"Cannot reach Matrix homeserver: {err}") from err


class MatrixChatClient:
    """Client wrapper for Matrix operations used by this integration."""

    def __init__(
        self,
        hass,
        session: ClientSession,
        entry_id: str,
        homeserver: str,
        user_id: str,
        password: str,
        access_token: str,
        verify_ssl: bool,
        encrypted_webhook_url: str,
        encrypted_webhook_token: str,
        dm_encrypted: bool,
        auto_convert_video: bool,
        video_convert_threshold_mb: float,
        max_upload_mb: float,
    ) -> None:
        self.hass = hass
        self._session = session
        self.entry_id = entry_id
        self.homeserver = homeserver.rstrip("/")
        self.user_id = user_id
        self.password = password
        self.access_token = _normalize_token(access_token)
        self.verify_ssl = verify_ssl
        self._ssl_arg = None if verify_ssl else False

        self.encrypted_webhook_url = (encrypted_webhook_url or "").strip()
        self.encrypted_webhook_token = (encrypted_webhook_token or "").strip()
        self.dm_encrypted = bool(dm_encrypted)

        self.auto_convert_video = bool(auto_convert_video)
        self.video_convert_threshold_mb = float(video_convert_threshold_mb)
        self.max_upload_mb = float(max_upload_mb)

        self.device_id = ""
        self._server_upload_limit_bytes: int | None = None

        self._store = Store(hass, 1, f"{DOMAIN}_{entry_id}_cache")
        self._dm_rooms: dict[str, str] = {}
        self._encrypted_rooms: dict[str, bool] = {}

    def _gateway_base_urls(self) -> list[str]:
        urls: list[str] = []
        # Prefer Docker-network endpoint for HA Container deployments.
        fallback = "http://matrix-e2ee-gateway:8080"
        urls.append(fallback)

        configured = (self.encrypted_webhook_url or "").rstrip("/")
        if configured and configured not in urls:
            urls.append(configured)
        return urls

    async def async_initialize(self) -> None:
        """Load cache and verify credentials."""
        cache = await self._store.async_load() or {}
        self._dm_rooms = dict(cache.get("dm_rooms") or {})
        self._encrypted_rooms = dict(cache.get("encrypted_rooms") or {})

        auth = await async_validate_credentials(
            session=self._session,
            homeserver=self.homeserver,
            user_id=self.user_id,
            password=self.password,
            access_token=self.access_token,
            verify_ssl=self.verify_ssl,
        )
        self.user_id = auth["user_id"]
        self.access_token = auth["access_token"]
        self.device_id = auth.get("device_id", "")

        # One-time best-effort migration: if an unencrypted placeholder DM already exists,
        # prefer upgrading it to E2EE and reusing it to avoid duplicate DMs.
        await self._async_migrate_placeholder_dms()

    async def _async_migrate_placeholder_dms(self) -> None:
        if not self.dm_encrypted or not self._dm_rooms:
            return

        changed = False
        for target_user, cached_room_id in list(self._dm_rooms.items()):
            if not isinstance(target_user, str) or not target_user.startswith("@"):
                continue
            discovered = await self._find_existing_dm_like_room(target_user)
            if not discovered or discovered == cached_room_id:
                continue

            if not await self._is_room_encrypted(discovered):
                try:
                    await self._enable_room_encryption(discovered)
                except Exception as err:  # noqa: BLE001
                    _LOGGER.warning(
                        "Could not enable encryption for existing DM room %s: %s",
                        discovered,
                        err,
                    )
                    continue

            if not await self._is_room_encrypted(discovered):
                continue

            try:
                await self._prefer_direct_room(target_user, discovered)
            except Exception as err:  # noqa: BLE001
                _LOGGER.warning("Could not update m.direct preference for %s: %s", target_user, err)

            self._dm_rooms[target_user] = discovered
            changed = True

        if changed:
            await self.async_persist_cache()

    async def async_persist_cache(self) -> None:
        await self._store.async_save(
            {
                "dm_rooms": self._dm_rooms,
                "encrypted_rooms": self._encrypted_rooms,
            }
        )

    async def async_get_gateway_health(self) -> dict[str, Any]:
        """Fetch /health from the encrypted gateway (best-effort)."""
        errors: list[str] = []
        for base_url in self._gateway_base_urls():
            url = f"{base_url}/health"
            try:
                async with self._session.get(url, ssl=self._ssl_arg) as resp:
                    text = await resp.text()
                    if 200 <= resp.status < 300:
                        try:
                            data = json.loads(text) if text else {}
                        except json.JSONDecodeError:
                            data = {"raw": text}
                        if not isinstance(data, dict):
                            data = {"raw": data}
                        return data
                    errors.append(f"{url} -> HTTP {resp.status}: {text[:200]}")
            except ClientError as err:
                errors.append(f"{url} -> {err}")

        raise MatrixChatConnectionError(
            f"Gateway health check failed: {' | '.join(errors) or 'no endpoints'}"
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        auth: bool = True,
        expected: tuple[int, ...] = (200,),
        allow_404: bool = False,
        data: Any = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        url = f"{self.homeserver}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"

        req_headers: dict[str, str] = dict(headers or {})
        if auth:
            req_headers["Authorization"] = f"Bearer {self.access_token}"

        for attempt in range(3):
            try:
                async with self._session.request(
                    method,
                    url,
                    json=payload,
                    data=data,
                    headers=req_headers,
                    ssl=self._ssl_arg,
                ) as resp:
                    text = await resp.text()

                    if allow_404 and resp.status == 404:
                        return None

                    if resp.status == 429 and attempt < 2:
                        retry_ms = 1000
                        try:
                            retry_data = json.loads(text)
                            retry_ms = int(retry_data.get("retry_after_ms", 1000))
                        except Exception:
                            retry_ms = 1000
                        await asyncio.sleep(max(0.2, retry_ms / 1000))
                        continue

                    if resp.status in expected:
                        if not text:
                            return {}
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return {"raw": text}

                    if resp.status in (401, 403):
                        raise MatrixChatAuthError(
                            f"Matrix auth failed HTTP {resp.status}: {text[:400]}"
                        )

                    raise MatrixChatError(
                        f"Matrix request failed {method} {path} HTTP {resp.status}: {text[:500]}"
                    )
            except ClientError as err:
                if attempt == 2:
                    raise MatrixChatConnectionError(
                        f"Connection problem during {method} {path}: {err}"
                    ) from err
                await asyncio.sleep(0.5)

        raise MatrixChatConnectionError(f"Retries exhausted for {method} {path}")

    def _next_txn_id(self) -> str:
        return f"ha-{time.time_ns()}-{uuid.uuid4().hex[:8]}"

    async def _resolve_alias(self, room_alias: str) -> str:
        encoded = urllib.parse.quote(room_alias, safe="")
        data = await self._request_json(
            "GET",
            f"/_matrix/client/v3/directory/room/{encoded}",
            expected=(200,),
        )
        room_id = (data or {}).get("room_id", "")
        if not room_id:
            raise MatrixChatError(f"Could not resolve room alias: {room_alias}")
        return room_id

    async def _get_direct_rooms_map(self) -> dict[str, list[str]]:
        user_enc = urllib.parse.quote(self.user_id, safe="")
        data = await self._request_json(
            "GET",
            f"/_matrix/client/v3/user/{user_enc}/account_data/m.direct",
            expected=(200,),
            allow_404=True,
        )
        if not isinstance(data, dict):
            return {}

        out: dict[str, list[str]] = {}
        for user, rooms in data.items():
            if isinstance(user, str) and isinstance(rooms, list):
                out[user] = [room for room in rooms if isinstance(room, str)]
        return out

    async def _get_joined_rooms(self) -> list[str]:
        data = await self._request_json(
            "GET",
            "/_matrix/client/v3/joined_rooms",
            expected=(200,),
        )
        rooms = (data or {}).get("joined_rooms")
        if not isinstance(rooms, list):
            return []
        return [r for r in rooms if isinstance(r, str) and r.startswith("!")]

    async def _get_room_name(self, room_id: str) -> str:
        encoded = urllib.parse.quote(room_id, safe="")
        data = await self._request_json(
            "GET",
            f"/_matrix/client/v3/rooms/{encoded}/state/m.room.name",
            expected=(200,),
            allow_404=True,
        )
        if not isinstance(data, dict):
            return ""
        name = data.get("name")
        return name if isinstance(name, str) else ""

    async def _get_room_joined_members(self, room_id: str) -> list[str]:
        encoded = urllib.parse.quote(room_id, safe="")
        data = await self._request_json(
            "GET",
            f"/_matrix/client/v3/rooms/{encoded}/joined_members",
            expected=(200,),
        )
        if not isinstance(data, dict):
            return []
        joined = data.get("joined") or {}
        if not isinstance(joined, dict):
            return []
        return [u for u in joined.keys() if isinstance(u, str) and u.startswith("@")]

    async def _enable_room_encryption(self, room_id: str) -> None:
        """Enable E2EE in an existing room (one-way operation)."""
        encoded = urllib.parse.quote(room_id, safe="")
        payload = {"algorithm": "m.megolm.v1.aes-sha2"}
        # state_key is the empty string; Matrix API expresses that as a trailing slash.
        await self._request_json(
            "PUT",
            f"/_matrix/client/v3/rooms/{encoded}/state/m.room.encryption/",
            payload=payload,
            expected=(200,),
        )
        # Avoid stale cache (we may have cached the room as unencrypted before).
        self._encrypted_rooms[room_id] = True

    async def _find_existing_dm_like_room(self, target_user: str) -> str | None:
        """Best-effort discovery of an existing DM room not present in m.direct.

        We only accept rooms whose name matches the Element-style placeholder:
        'DM with @user:server'. This prevents accidentally selecting a normal room
        that happens to have 2 members.
        """
        expected_name = f"DM with {target_user}"
        for room_id in await self._get_joined_rooms():
            try:
                name = await self._get_room_name(room_id)
                if name != expected_name:
                    continue
                members = await self._get_room_joined_members(room_id)
                if set(members) == {self.user_id, target_user}:
                    return room_id
            except Exception:  # noqa: BLE001
                continue
        return None

    async def _prefer_direct_room(self, target_user: str, preferred_room_id: str) -> None:
        direct_map = await self._get_direct_rooms_map()
        existing = direct_map.get(target_user, [])
        merged = [preferred_room_id] + [r for r in existing if r != preferred_room_id]
        direct_map[target_user] = merged
        await self._put_direct_rooms_map(direct_map)

    async def _put_direct_rooms_map(self, mapping: dict[str, list[str]]) -> None:
        user_enc = urllib.parse.quote(self.user_id, safe="")
        await self._request_json(
            "PUT",
            f"/_matrix/client/v3/user/{user_enc}/account_data/m.direct",
            payload=mapping,
            expected=(200,),
        )

    async def _create_dm_room(self, target_user: str, encrypted: bool) -> str:
        payload: dict[str, Any] = {
            "is_direct": True,
            "invite": [target_user],
            "preset": "private_chat",
        }
        if encrypted:
            payload["initial_state"] = [
                {
                    "type": "m.room.encryption",
                    "state_key": "",
                    "content": {"algorithm": "m.megolm.v1.aes-sha2"},
                }
            ]

        data = await self._request_json(
            "POST",
            "/_matrix/client/v3/createRoom",
            payload=payload,
            expected=(200,),
        )
        room_id = (data or {}).get("room_id", "")
        if not room_id:
            raise MatrixChatError(f"Could not create DM room for {target_user}")
        self._encrypted_rooms[room_id] = encrypted
        return room_id

    async def _pick_direct_room(self, room_ids: list[str]) -> str | None:
        if not room_ids:
            return None

        encrypted_candidates: list[str] = []
        plain_candidates: list[str] = []
        for room_id in room_ids:
            try:
                if await self._is_room_encrypted(room_id):
                    encrypted_candidates.append(room_id)
                else:
                    plain_candidates.append(room_id)
            except Exception:
                plain_candidates.append(room_id)

        if self.dm_encrypted:
            if encrypted_candidates:
                return encrypted_candidates[0]
            return None

        if encrypted_candidates:
            return encrypted_candidates[0]
        if plain_candidates:
            return plain_candidates[0]
        return None

    async def _resolve_dm_room(self, target_user: str) -> str:
        cached = self._dm_rooms.get(target_user)
        if cached:
            if self.dm_encrypted:
                if await self._is_room_encrypted(cached):
                    return cached
            else:
                return cached

        direct_map = await self._get_direct_rooms_map()
        room_ids = direct_map.get(target_user, [])
        selected = await self._pick_direct_room(room_ids)
        if selected:
            self._dm_rooms[target_user] = selected
            await self.async_persist_cache()
            return selected

        # No usable entry in m.direct: try best-effort discovery of an existing placeholder DM.
        discovered = await self._find_existing_dm_like_room(target_user)
        if discovered:
            if self.dm_encrypted and not await self._is_room_encrypted(discovered):
                try:
                    await self._enable_room_encryption(discovered)
                except Exception as err:  # noqa: BLE001
                    _LOGGER.warning(
                        "Could not enable encryption for existing DM room %s: %s",
                        discovered,
                        err,
                    )
            if not self.dm_encrypted or await self._is_room_encrypted(discovered):
                await self._prefer_direct_room(target_user, discovered)
                self._dm_rooms[target_user] = discovered
                await self.async_persist_cache()
                return discovered

        room_id = await self._create_dm_room(target_user, encrypted=self.dm_encrypted)
        merged = [room_id] + [r for r in room_ids if r != room_id]
        direct_map[target_user] = merged
        await self._put_direct_rooms_map(direct_map)

        self._dm_rooms[target_user] = room_id
        await self.async_persist_cache()
        return room_id

    async def _is_room_encrypted(self, room_id: str) -> bool:
        if room_id in self._encrypted_rooms:
            return self._encrypted_rooms[room_id]

        encoded = urllib.parse.quote(room_id, safe="")
        data = await self._request_json(
            "GET",
            f"/_matrix/client/v3/rooms/{encoded}/state/m.room.encryption",
            expected=(200,),
            allow_404=True,
        )
        encrypted = data is not None
        self._encrypted_rooms[room_id] = encrypted
        await self.async_persist_cache()
        return encrypted

    async def _resolve_target_to_room(self, target: str) -> tuple[str, str]:
        target = target.strip()
        if not target:
            raise MatrixChatError("Empty target provided")

        if target.startswith("@"):
            room_id = await self._resolve_dm_room(target)
            return room_id, "user_dm"
        if target.startswith("!"):
            return target, "room_id"
        if target.startswith("#"):
            room_id = await self._resolve_alias(target)
            return room_id, "room_alias"

        raise MatrixChatError(
            f"Target '{target}' is invalid. Use @user:server, !room:server or #alias:server"
        )

    async def _send_room_event(
        self, room_id: str, message_type: str, content: dict[str, Any]
    ) -> str:
        encoded_room = urllib.parse.quote(room_id, safe="")
        encoded_type = urllib.parse.quote(message_type, safe="")
        txn_id = self._next_txn_id()
        data = await self._request_json(
            "PUT",
            f"/_matrix/client/v3/rooms/{encoded_room}/send/{encoded_type}/{txn_id}",
            payload=content,
            expected=(200,),
        )
        event_id = (data or {}).get("event_id", "")
        if not event_id:
            raise MatrixChatError(f"Matrix did not return event_id for room {room_id}")
        return event_id

    async def _send_encrypted_gateway_text(
        self,
        room_id: str,
        message: str,
        message_format: str,
        reply_to_event_id: str = "",
        edit_event_id: str = "",
    ) -> str:
        if not self.encrypted_webhook_url and not self._gateway_base_urls():
            raise MatrixChatError(
                f"Room {room_id} is encrypted and no encrypted gateway URL is configured"
            )

        headers = {"Content-Type": "application/json"}
        if self.encrypted_webhook_token:
            headers["Authorization"] = f"Bearer {self.encrypted_webhook_token}"

        payload = {
            "room_id": room_id,
            "message": message,
            "format": message_format,
            "reply_to_event_id": reply_to_event_id,
            "edit_event_id": edit_event_id,
        }

        errors: list[str] = []
        for base_url in self._gateway_base_urls():
            url = f"{base_url}/send_text"
            try:
                async with self._session.post(
                    url,
                    json=payload,
                    headers=headers,
                    ssl=self._ssl_arg,
                ) as resp:
                    text = await resp.text()
                    if 200 <= resp.status < 300:
                        try:
                            data = json.loads(text) if text else {}
                        except json.JSONDecodeError:
                            data = {}
                        return data.get("event_id", "")
                    errors.append(f"{url} -> HTTP {resp.status}: {text[:250]}")
            except ClientError as err:
                errors.append(f"{url} -> {err}")

        raise MatrixChatError(f"Encrypted gateway text failed: {' | '.join(errors)}")

    async def _send_encrypted_gateway_reaction(
        self, room_id: str, event_id: str, reaction_key: str
    ) -> str:
        if not self.encrypted_webhook_url and not self._gateway_base_urls():
            raise MatrixChatError(
                f"Room {room_id} is encrypted and no encrypted gateway URL is configured"
            )

        headers = {"Content-Type": "application/json"}
        if self.encrypted_webhook_token:
            headers["Authorization"] = f"Bearer {self.encrypted_webhook_token}"

        payload = {
            "room_id": room_id,
            "event_id": event_id,
            "reaction_key": reaction_key,
        }

        errors: list[str] = []
        for base_url in self._gateway_base_urls():
            url = f"{base_url}/send_reaction"
            try:
                async with self._session.post(
                    url,
                    json=payload,
                    headers=headers,
                    ssl=self._ssl_arg,
                ) as resp:
                    text = await resp.text()
                    if 200 <= resp.status < 300:
                        try:
                            data = json.loads(text) if text else {}
                        except json.JSONDecodeError:
                            data = {}
                        return data.get("event_id", "")
                    errors.append(f"{url} -> HTTP {resp.status}: {text[:250]}")
            except ClientError as err:
                errors.append(f"{url} -> {err}")

        raise MatrixChatError(f"Encrypted gateway reaction failed: {' | '.join(errors)}")

    async def _send_encrypted_gateway_media(
        self,
        room_id: str,
        file_path: Path,
        mime_type: str,
        msgtype: str,
        body: str,
        caption: str,
        info: dict[str, Any],
    ) -> str:
        if not self.encrypted_webhook_url and not self._gateway_base_urls():
            raise MatrixChatError(
                f"Room {room_id} is encrypted and no encrypted gateway URL is configured"
            )

        headers: dict[str, str] = {}
        if self.encrypted_webhook_token:
            headers["Authorization"] = f"Bearer {self.encrypted_webhook_token}"

        file_bytes = await self.hass.async_add_executor_job(file_path.read_bytes)
        errors: list[str] = []
        for base_url in self._gateway_base_urls():
            form = FormData()
            form.add_field("room_id", room_id)
            form.add_field("msgtype", msgtype)
            form.add_field("body", body)
            form.add_field("caption", caption or "")
            form.add_field("mime_type", mime_type)
            form.add_field("info", json.dumps(info))
            form.add_field(
                "file",
                file_bytes,
                filename=file_path.name,
                content_type=mime_type,
            )
            url = f"{base_url}/send_media"
            try:
                async with self._session.post(
                    url,
                    data=form,
                    headers=headers,
                    ssl=self._ssl_arg,
                ) as resp:
                    text = await resp.text()
                    if 200 <= resp.status < 300:
                        try:
                            data = json.loads(text) if text else {}
                        except json.JSONDecodeError:
                            data = {}
                        event_id = data.get("event_id", "")
                        if not event_id:
                            raise MatrixChatError(
                                "Encrypted gateway media response missing event_id"
                            )
                        return event_id
                    errors.append(f"{url} -> HTTP {resp.status}: {text[:250]}")
            except ClientError as err:
                errors.append(f"{url} -> {err}")

        raise MatrixChatError(f"Encrypted gateway media failed: {' | '.join(errors)}")

    def _build_message_content(
        self,
        message: str,
        message_format: str,
        reply_to_event_id: str,
        edit_event_id: str,
    ) -> dict[str, Any]:
        if reply_to_event_id and edit_event_id:
            raise MatrixChatError("reply_to_event_id and edit_event_id are mutually exclusive")

        base_content: dict[str, Any] = {"msgtype": "m.text", "body": message}
        if message_format == FORMAT_HTML:
            base_content["format"] = "org.matrix.custom.html"
            base_content["formatted_body"] = message

        if reply_to_event_id:
            base_content["m.relates_to"] = {
                "m.in_reply_to": {"event_id": reply_to_event_id}
            }
            return base_content

        if edit_event_id:
            edit_content = {
                "msgtype": "m.text",
                "body": f"* {message}",
                "m.new_content": dict(base_content),
                "m.relates_to": {
                    "rel_type": "m.replace",
                    "event_id": edit_event_id,
                },
            }
            if message_format == FORMAT_HTML:
                edit_content["format"] = "org.matrix.custom.html"
                edit_content["formatted_body"] = f"* {message}"
            return edit_content

        return base_content

    async def async_send_message(
        self,
        targets: list[str],
        message: str,
        message_format: str,
        reply_to_event_id: str = "",
        edit_event_id: str = "",
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        content = self._build_message_content(
            message=message,
            message_format=message_format,
            reply_to_event_id=reply_to_event_id,
            edit_event_id=edit_event_id,
        )

        for target in targets:
            try:
                room_id, target_type = await self._resolve_target_to_room(target)
                encrypted = await self._is_room_encrypted(room_id)

                if encrypted:
                    event_id = await self._send_encrypted_gateway_text(
                        room_id=room_id,
                        message=message,
                        message_format=message_format,
                        reply_to_event_id=reply_to_event_id,
                        edit_event_id=edit_event_id,
                    )
                    results.append(
                        {
                            "target": target,
                            "target_type": target_type,
                            "room_id": room_id,
                            "event_id": event_id,
                            "transport": "encrypted_gateway",
                            "encrypted": True,
                            "status": "sent",
                        }
                    )
                    continue

                event_id = await self._send_room_event(room_id, "m.room.message", content)
                results.append(
                    {
                        "target": target,
                        "target_type": target_type,
                        "room_id": room_id,
                        "event_id": event_id,
                        "transport": "direct",
                        "encrypted": False,
                        "status": "sent",
                    }
                )
            except Exception as err:  # noqa: BLE001
                _LOGGER.exception("Failed sending message to target %s", target)
                results.append(
                    {
                        "target": target,
                        "status": "failed",
                        "error": str(err),
                    }
                )

        success_count = sum(1 for item in results if item.get("status") == "sent")
        return {
            "results": results,
            "success_count": success_count,
            "failure_count": len(results) - success_count,
        }

    async def async_send_reaction(
        self,
        targets: list[str],
        event_id: str,
        reaction_key: str,
    ) -> dict[str, Any]:
        content: dict[str, Any] = {
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": event_id,
                "key": reaction_key,
            }
        }

        results: list[dict[str, Any]] = []
        for target in targets:
            try:
                room_id, target_type = await self._resolve_target_to_room(target)
                encrypted = await self._is_room_encrypted(room_id)

                if encrypted:
                    reaction_event_id = await self._send_encrypted_gateway_reaction(
                        room_id=room_id,
                        event_id=event_id,
                        reaction_key=reaction_key,
                    )
                    results.append(
                        {
                            "target": target,
                            "target_type": target_type,
                            "room_id": room_id,
                            "event_id": reaction_event_id,
                            "transport": "encrypted_gateway",
                            "encrypted": True,
                            "status": "sent",
                        }
                    )
                    continue

                reaction_event_id = await self._send_room_event(
                    room_id, "m.reaction", content
                )
                results.append(
                    {
                        "target": target,
                        "target_type": target_type,
                        "room_id": room_id,
                        "event_id": reaction_event_id,
                        "transport": "direct",
                        "encrypted": False,
                        "status": "sent",
                    }
                )
            except Exception as err:  # noqa: BLE001
                _LOGGER.exception("Failed sending reaction to target %s", target)
                results.append(
                    {
                        "target": target,
                        "status": "failed",
                        "error": str(err),
                    }
                )

        success_count = sum(1 for item in results if item.get("status") == "sent")
        return {
            "results": results,
            "success_count": success_count,
            "failure_count": len(results) - success_count,
        }

    async def _get_server_upload_limit_bytes(self) -> int | None:
        if self._server_upload_limit_bytes is not None:
            return self._server_upload_limit_bytes

        data = await self._request_json(
            "GET",
            "/_matrix/media/v3/config",
            expected=(200,),
        )
        value = (data or {}).get("m.upload.size")
        if isinstance(value, int) and value > 0:
            self._server_upload_limit_bytes = value
        else:
            self._server_upload_limit_bytes = None
        return self._server_upload_limit_bytes

    async def _ensure_upload_size_allowed(self, path: Path, max_size_mb: float) -> None:
        size = path.stat().st_size
        local_limit = int(max_size_mb * 1024 * 1024)
        server_limit = await self._get_server_upload_limit_bytes()
        effective_limit = min(local_limit, server_limit) if server_limit else local_limit

        if size > effective_limit:
            raise MatrixChatError(
                f"File too large ({size} bytes). Effective upload limit is {effective_limit} bytes"
            )

    async def _convert_video_if_needed(
        self,
        file_path: Path,
        mime_type: str,
        auto_convert: bool,
        convert_threshold_mb: float,
    ) -> tuple[Path, str, bool]:
        if not mime_type.startswith("video/"):
            return file_path, mime_type, False

        size = file_path.stat().st_size
        threshold_bytes = int(convert_threshold_mb * 1024 * 1024)
        if size <= threshold_bytes or not auto_convert:
            return file_path, mime_type, False

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise MatrixChatError("ffmpeg not found but video conversion is required")

        output_path = Path(tempfile.gettempdir()) / f"matrix_chat_{uuid.uuid4().hex}.mp4"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(file_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise MatrixChatError(
                f"Video conversion failed with ffmpeg exit {proc.returncode}: {stderr.decode(errors='ignore')[:600]}"
            )

        return output_path, "video/mp4", True

    async def _upload_media(self, path: Path, mime_type: str) -> str:
        encoded_filename = urllib.parse.quote(path.name)
        upload_path = f"/_matrix/media/v3/upload?filename={encoded_filename}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": mime_type,
        }

        url = f"{self.homeserver}{upload_path}"
        file_bytes = await self.hass.async_add_executor_job(path.read_bytes)
        async with self._session.post(
            url,
            headers=headers,
            data=file_bytes,
            ssl=self._ssl_arg,
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise MatrixChatError(
                    f"Media upload failed HTTP {resp.status}: {text[:500]}"
                )
            data = json.loads(text)

        content_uri = data.get("content_uri", "")
        if not content_uri:
            raise MatrixChatError("Matrix upload response did not contain content_uri")
        return content_uri

    async def _build_media_info(self, path: Path, mime_type: str) -> dict[str, Any]:
        info: dict[str, Any] = {
            "mimetype": mime_type,
            "size": path.stat().st_size,
        }

        if mime_type.startswith("image/"):
            try:
                width, height = await self.hass.async_add_executor_job(
                    _image_dimensions, path
                )
                info["w"] = width
                info["h"] = height
            except Exception:  # noqa: BLE001
                _LOGGER.debug("Could not determine image dimensions for %s", path)

        if mime_type.startswith("video/"):
            ffprobe_bin = shutil.which("ffprobe")
            if ffprobe_bin:
                cmd = [
                    ffprobe_bin,
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=width,height:format=duration",
                    "-of",
                    "json",
                    str(path),
                ]
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, _stderr = await proc.communicate()
                    if proc.returncode == 0 and stdout:
                        probe = json.loads(stdout.decode(errors="ignore"))
                        streams = probe.get("streams") or []
                        for stream in streams:
                            width = stream.get("width")
                            height = stream.get("height")
                            if isinstance(width, int) and width > 0:
                                info["w"] = width
                            if isinstance(height, int) and height > 0:
                                info["h"] = height
                            if "w" in info and "h" in info:
                                break
                        duration = (probe.get("format") or {}).get("duration")
                        if duration is not None:
                            try:
                                info["duration"] = int(float(duration) * 1000)
                            except (TypeError, ValueError):
                                pass
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("Could not determine video metadata for %s", path)

        return info

    async def async_send_media(
        self,
        targets: list[str],
        file_path: str,
        message: str,
        mime_type: str,
        auto_convert: bool,
        convert_threshold_mb: float,
        max_size_mb: float,
    ) -> dict[str, Any]:
        path = Path(file_path)
        if not path.is_file():
            raise MatrixChatError(f"File path does not exist: {file_path}")

        allowed = await self.hass.async_add_executor_job(
            self.hass.config.is_allowed_path, str(path)
        )
        if not allowed:
            raise MatrixChatError(
                f"Path is not allowed by Home Assistant allowlist: {file_path}"
            )

        detected_mime = mime_type or (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        sniffed = await self.hass.async_add_executor_job(_sniff_mime_type, path)
        if sniffed and sniffed != detected_mime:
            _LOGGER.warning(
                "Sniffed MIME type %s differs from %s for %s; using sniffed type",
                sniffed,
                detected_mime,
                path,
            )
            detected_mime = sniffed
        prepared_path = path
        prepared_mime = detected_mime
        cleanup_prepared = False

        prepared_path, prepared_mime, cleanup_prepared = await self._convert_video_if_needed(
            file_path=path,
            mime_type=detected_mime,
            auto_convert=auto_convert,
            convert_threshold_mb=convert_threshold_mb,
        )

        await self._ensure_upload_size_allowed(prepared_path, max_size_mb)
        info = await self._build_media_info(prepared_path, prepared_mime)
        msgtype = _media_msgtype(prepared_mime)
        media_body = prepared_path.name

        resolved_targets: list[dict[str, Any]] = []
        for target in targets:
            room_id, target_type = await self._resolve_target_to_room(target)
            encrypted = await self._is_room_encrypted(room_id)
            resolved_targets.append(
                {
                    "target": target,
                    "target_type": target_type,
                    "room_id": room_id,
                    "encrypted": encrypted,
                }
            )

        direct_targets = [t for t in resolved_targets if not t["encrypted"]]
        encrypted_targets = [t for t in resolved_targets if t["encrypted"]]

        content_uri = ""
        if direct_targets:
            content_uri = await self._upload_media(prepared_path, prepared_mime)

        results: list[dict[str, Any]] = []

        for item in resolved_targets:
            target = item["target"]
            room_id = item["room_id"]
            target_type = item["target_type"]
            encrypted = item["encrypted"]
            try:
                if encrypted:
                    event_id = await self._send_encrypted_gateway_media(
                        room_id=room_id,
                        file_path=prepared_path,
                        mime_type=prepared_mime,
                        msgtype=msgtype,
                        body=media_body,
                        caption=message,
                        info=info,
                    )
                    results.append(
                        {
                            "target": target,
                            "target_type": target_type,
                            "room_id": room_id,
                            "event_id": event_id,
                            "transport": "encrypted_gateway",
                            "encrypted": True,
                            "status": "sent",
                            "msgtype": msgtype,
                        }
                    )
                    continue

                if message:
                    await self._send_room_event(
                        room_id,
                        "m.room.message",
                        {"msgtype": "m.text", "body": message},
                    )

                content = {
                    "msgtype": msgtype,
                    "body": media_body,
                    "filename": media_body,
                    "url": content_uri,
                    "info": info,
                }
                event_id = await self._send_room_event(room_id, "m.room.message", content)
                results.append(
                    {
                        "target": target,
                        "target_type": target_type,
                        "room_id": room_id,
                        "event_id": event_id,
                        "transport": "direct",
                        "encrypted": False,
                        "status": "sent",
                        "msgtype": msgtype,
                    }
                )
            except Exception as err:  # noqa: BLE001
                _LOGGER.exception("Failed sending media to target %s", target)
                results.append(
                    {
                        "target": target,
                        "status": "failed",
                        "error": str(err),
                    }
                )

        if cleanup_prepared:
            try:
                prepared_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                _LOGGER.warning("Could not cleanup temporary file %s", prepared_path)

        success_count = sum(1 for item in results if item.get("status") == "sent")
        response: dict[str, Any] = {
            "results": results,
            "success_count": success_count,
            "failure_count": len(results) - success_count,
        }
        if content_uri:
            response["content_uri"] = content_uri
        return response
