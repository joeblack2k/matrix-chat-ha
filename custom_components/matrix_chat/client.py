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

_OUTBOX_MAX_ITEMS = 1000


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

        self._outbox_store = Store(hass, 1, f"{DOMAIN}_{entry_id}_outbox")
        self._outbox: list[dict[str, Any]] = []
        self._outbox_lock = asyncio.Lock()
        self._outbox_last_error: str = ""

    @staticmethod
    def _is_transient_error(err: Exception) -> bool:
        # Only queue on clear connectivity issues (gateway down / homeserver unreachable).
        if isinstance(err, MatrixChatConnectionError):
            return True
        return False

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
        await self._async_outbox_load()

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

    async def _async_outbox_load(self) -> None:
        data = await self._outbox_store.async_load() or {}
        items = data.get("items") if isinstance(data, dict) else None
        if isinstance(items, list):
            # Best-effort validation; ignore malformed entries.
            loaded: list[dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if not isinstance(item.get("id"), str) or not item["id"]:
                    continue
                if item.get("kind") not in {"message", "media", "reaction", "redaction"}:
                    continue
                if not isinstance(item.get("payload"), dict):
                    continue
                loaded.append(item)
            self._outbox = loaded[-_OUTBOX_MAX_ITEMS:]
        else:
            self._outbox = []
        self._outbox_last_error = str((data or {}).get("last_error") or "")

    async def _async_outbox_save(self) -> None:
        await self._outbox_store.async_save(
            {
                "items": self._outbox[-_OUTBOX_MAX_ITEMS:],
                "last_error": self._outbox_last_error,
            }
        )

    async def async_get_outbox_stats(self) -> dict[str, Any]:
        async with self._outbox_lock:
            size = len(self._outbox)
            oldest = None
            if self._outbox:
                try:
                    oldest = float(self._outbox[0].get("created_ts") or 0.0)
                except (TypeError, ValueError):
                    oldest = None
            # If there's nothing queued, don't keep surfacing a stale error in UI.
            last_error = self._outbox_last_error if size else ""
            return {
                "outbox_size": size,
                "outbox_oldest_ts": oldest,
                "outbox_last_error": last_error or "",
            }

    async def _async_outbox_enqueue(self, *, kind: str, payload: dict[str, Any], error: str) -> str:
        item_id = uuid.uuid4().hex
        now = time.time()
        item = {
            "id": item_id,
            "kind": kind,
            "payload": payload,
            "created_ts": now,
            "attempts": 0,
            "next_attempt_ts": now,
            "last_error": str(error)[:500],
        }
        async with self._outbox_lock:
            self._outbox.append(item)
            # Bound size to avoid unbounded growth.
            if len(self._outbox) > _OUTBOX_MAX_ITEMS:
                self._outbox = self._outbox[-_OUTBOX_MAX_ITEMS:]
            self._outbox_last_error = str(error)[:500]
            await self._async_outbox_save()
        return item_id

    async def async_flush_outbox(self, *, max_items: int = 25) -> dict[str, Any]:
        """Try sending queued outbound items. Returns stats about this flush attempt."""
        sent = 0
        failed = 0
        now = time.time()

        for _ in range(max(0, int(max_items))):
            async with self._outbox_lock:
                # Find first due item (stable order).
                idx = None
                for i, item in enumerate(self._outbox):
                    try:
                        due = float(item.get("next_attempt_ts") or 0.0)
                    except (TypeError, ValueError):
                        due = 0.0
                    if due <= now:
                        idx = i
                        break
                if idx is None:
                    break
                item = self._outbox.pop(idx)
                await self._async_outbox_save()

            kind = str(item.get("kind") or "")
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            try:
                send_result: dict[str, Any] | None = None
                if kind == "message":
                    send_result = await self.async_send_message(  # type: ignore[arg-type]
                        queue_on_fail=False, **payload
                    )
                elif kind == "media":
                    send_result = await self.async_send_media(  # type: ignore[arg-type]
                        queue_on_fail=False, **payload
                    )
                elif kind == "reaction":
                    send_result = await self.async_send_reaction(  # type: ignore[arg-type]
                        queue_on_fail=False, **payload
                    )
                elif kind == "redaction":
                    send_result = await self.async_redact_event(  # type: ignore[arg-type]
                        queue_on_fail=False, **payload
                    )
                else:
                    raise MatrixChatError(f"Unknown outbox kind: {kind}")

                success_count = int((send_result or {}).get("success_count") or 0)
                failure_count = int((send_result or {}).get("failure_count") or 0)
                if success_count <= 0 or failure_count > 0:
                    err_detail = ""
                    results = (send_result or {}).get("results")
                    if isinstance(results, list):
                        for result_item in results:
                            if not isinstance(result_item, dict):
                                continue
                            if result_item.get("status") == "sent":
                                continue
                            err_detail = str(
                                result_item.get("error")
                                or result_item.get("status")
                                or "send_failed"
                            )[:500]
                            if err_detail:
                                break
                    detail_suffix = f": {err_detail}" if err_detail else ""
                    raise MatrixChatError(
                        "Outbox replay failed for "
                        f"{kind} (success_count={success_count}, failure_count={failure_count})"
                        f"{detail_suffix}"
                    )
                sent += 1
                continue
            except Exception as err:  # noqa: BLE001
                failed += 1
                attempts = int(item.get("attempts") or 0) + 1
                backoff = min(3600.0, (2 ** min(attempts, 10)) * 5.0)  # 5s .. ~1h
                item["attempts"] = attempts
                item["next_attempt_ts"] = time.time() + backoff
                item["last_error"] = str(err)[:500]
                async with self._outbox_lock:
                    self._outbox.append(item)
                    self._outbox_last_error = str(err)[:500]
                    await self._async_outbox_save()

        stats = await self.async_get_outbox_stats()
        # If the queue is empty after this flush, clear last_error so it doesn't linger.
        if stats.get("outbox_size") in (0, None):
            async with self._outbox_lock:
                if self._outbox_last_error:
                    self._outbox_last_error = ""
                    await self._async_outbox_save()
            stats["outbox_last_error"] = ""
        stats.update({"flush_sent": sent, "flush_failed": failed})
        return stats

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
        await self.async_persist_cache()

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

        # If we have direct rooms for this user but none are encrypted, prefer upgrading
        # an existing DM to E2EE instead of creating a brand new room (avoids duplicates).
        if self.dm_encrypted and room_ids:
            for candidate in room_ids:
                try:
                    if not await self._is_room_encrypted(candidate):
                        await self._enable_room_encryption(candidate)
                    if await self._is_room_encrypted(candidate):
                        await self._prefer_direct_room(target_user, candidate)
                        self._dm_rooms[target_user] = candidate
                        await self.async_persist_cache()
                        return candidate
                except Exception as err:  # noqa: BLE001
                    _LOGGER.debug(
                        "Could not upgrade direct room %s to encrypted DM for %s: %s",
                        candidate,
                        target_user,
                        err,
                    )

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
        cached = self._encrypted_rooms.get(room_id)
        if cached is True:
            return True
        # Cached "False" can go stale (for example after external room changes or
        # older cache snapshots). Revalidate against live room state before deciding.

        encoded = urllib.parse.quote(room_id, safe="")
        # Matrix state events always have a state_key. For m.room.encryption the state_key
        # is the empty string, which is represented by a trailing slash in the client API.
        data = await self._request_json(
            "GET",
            f"/_matrix/client/v3/rooms/{encoded}/state/m.room.encryption/",
            expected=(200,),
            allow_404=True,
        )
        encrypted = data is not None
        if cached != encrypted or room_id not in self._encrypted_rooms:
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

    async def async_resolve_target(self, target: str) -> dict[str, Any]:
        """Resolve a target (@user, !room_id, #alias) to a concrete room_id."""
        room_id, target_type = await self._resolve_target_to_room(target)
        encrypted = await self._is_room_encrypted(room_id)
        return {
            "target": target,
            "target_type": target_type,
            "room_id": room_id,
            "encrypted": encrypted,
        }

    async def async_ensure_dm(self, user_id: str) -> str:
        """Ensure a DM room exists for a user and return the room_id."""
        user_id = str(user_id or "").strip()
        if not user_id.startswith("@"):
            raise MatrixChatError("user_id must be a Matrix user ID like @user:server")
        return await self._resolve_dm_room(user_id)

    async def async_ensure_room_encrypted(self, room_id: str) -> bool:
        """Ensure room has E2EE enabled (one-way). Returns whether it is encrypted."""
        room_id = str(room_id or "").strip()
        if not room_id.startswith("!"):
            raise MatrixChatError("room_id must be a Matrix room ID like !abc:server")
        if await self._is_room_encrypted(room_id):
            return True
        await self._enable_room_encryption(room_id)
        return await self._is_room_encrypted(room_id)

    async def async_join_room(self, room_or_alias: str) -> str:
        """Join a room by room_id or alias. Returns the joined room_id."""
        room_or_alias = str(room_or_alias or "").strip()
        if not room_or_alias or not (room_or_alias.startswith("!") or room_or_alias.startswith("#")):
            raise MatrixChatError("room_or_alias must start with ! (room_id) or # (alias)")

        encoded = urllib.parse.quote(room_or_alias, safe="")
        data = await self._request_json(
            "POST",
            f"/_matrix/client/v3/join/{encoded}",
            payload={},
            expected=(200,),
        )
        room_id = (data or {}).get("room_id", "")
        if isinstance(room_id, str) and room_id.startswith("!"):
            return room_id
        # Some servers may omit room_id in response for already-joined rooms; best-effort fallback.
        if room_or_alias.startswith("!"):
            return room_or_alias
        raise MatrixChatError(f"Join did not return room_id for {room_or_alias}")

    async def async_invite_user(self, room_id: str, user_id: str) -> None:
        """Invite a user to a room."""
        room_id = str(room_id or "").strip()
        user_id = str(user_id or "").strip()
        if not room_id.startswith("!"):
            raise MatrixChatError("room_id must be a Matrix room ID like !abc:server")
        if not user_id.startswith("@"):
            raise MatrixChatError("user_id must be a Matrix user ID like @user:server")

        encoded = urllib.parse.quote(room_id, safe="")
        await self._request_json(
            "POST",
            f"/_matrix/client/v3/rooms/{encoded}/invite",
            payload={"user_id": user_id},
            expected=(200,),
        )

    async def async_list_joined_rooms(
        self, *, limit: int = 50, include_members: bool = False
    ) -> list[dict[str, Any]]:
        """List rooms the bot is currently joined to (best-effort)."""
        try:
            limit_i = max(1, min(500, int(limit)))
        except (TypeError, ValueError):
            limit_i = 50

        rooms = await self._get_joined_rooms()
        out: list[dict[str, Any]] = []
        for room_id in rooms[:limit_i]:
            try:
                name = await self._get_room_name(room_id)
            except Exception:  # noqa: BLE001
                name = ""
            try:
                encrypted = await self._is_room_encrypted(room_id)
            except Exception:  # noqa: BLE001
                encrypted = False

            item: dict[str, Any] = {"room_id": room_id, "name": name, "encrypted": encrypted}
            if include_members:
                try:
                    item["members"] = await self._get_room_joined_members(room_id)
                except Exception:  # noqa: BLE001
                    item["members"] = []
            out.append(item)
        return out

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

    async def _redact_room_event(self, room_id: str, event_id: str, reason: str) -> str:
        encoded_room = urllib.parse.quote(room_id, safe="")
        encoded_event = urllib.parse.quote(event_id, safe="")
        txn_id = self._next_txn_id()
        payload: dict[str, Any] = {}
        if reason:
            payload["reason"] = reason
        data = await self._request_json(
            "PUT",
            f"/_matrix/client/v3/rooms/{encoded_room}/redact/{encoded_event}/{txn_id}",
            payload=payload,
            expected=(200,),
        )
        redaction_event_id = (data or {}).get("event_id", "")
        if not redaction_event_id:
            raise MatrixChatError(
                f"Matrix did not return redaction event_id for room {room_id}"
            )
        return redaction_event_id

    async def _send_encrypted_gateway_text(
        self,
        room_id: str,
        message: str,
        message_format: str,
        silent: bool = False,
        reply_to_event_id: str = "",
        edit_event_id: str = "",
        thread_root_event_id: str = "",
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
            "silent": bool(silent),
            "reply_to_event_id": reply_to_event_id,
            "edit_event_id": edit_event_id,
            "thread_root_event_id": thread_root_event_id,
        }

        errors: list[str] = []
        transient = False
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
                    if resp.status >= 500:
                        transient = True
                    errors.append(f"{url} -> HTTP {resp.status}: {text[:250]}")
            except ClientError as err:
                transient = True
                errors.append(f"{url} -> {err}")

        msg = f"Encrypted gateway text failed: {' | '.join(errors)}"
        if transient:
            raise MatrixChatConnectionError(msg)
        raise MatrixChatError(msg)

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
        transient = False
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
                    if resp.status >= 500:
                        transient = True
                    errors.append(f"{url} -> HTTP {resp.status}: {text[:250]}")
            except ClientError as err:
                transient = True
                errors.append(f"{url} -> {err}")

        msg = f"Encrypted gateway reaction failed: {' | '.join(errors)}"
        if transient:
            raise MatrixChatConnectionError(msg)
        raise MatrixChatError(msg)

    async def _send_encrypted_gateway_media(
        self,
        room_id: str,
        file_path: Path,
        mime_type: str,
        msgtype: str,
        body: str,
        caption: str,
        info: dict[str, Any],
        thread_root_event_id: str = "",
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
        transient = False
        for base_url in self._gateway_base_urls():
            form = FormData()
            form.add_field("room_id", room_id)
            form.add_field("msgtype", msgtype)
            form.add_field("body", body)
            form.add_field("caption", caption or "")
            form.add_field("mime_type", mime_type)
            form.add_field("info", json.dumps(info))
            form.add_field("thread_root_event_id", str(thread_root_event_id or ""))
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
                    if resp.status >= 500:
                        transient = True
                    errors.append(f"{url} -> HTTP {resp.status}: {text[:250]}")
            except ClientError as err:
                transient = True
                errors.append(f"{url} -> {err}")

        msg = f"Encrypted gateway media failed: {' | '.join(errors)}"
        if transient:
            raise MatrixChatConnectionError(msg)
        raise MatrixChatError(msg)

    def _build_message_content(
        self,
        message: str,
        message_format: str,
        silent: bool,
        reply_to_event_id: str,
        edit_event_id: str,
        thread_root_event_id: str,
    ) -> dict[str, Any]:
        thread_root_event_id = str(thread_root_event_id or "").strip()
        if reply_to_event_id and edit_event_id:
            raise MatrixChatError("reply_to_event_id and edit_event_id are mutually exclusive")
        if thread_root_event_id and edit_event_id:
            raise MatrixChatError("thread_root_event_id and edit_event_id are mutually exclusive")

        msgtype = "m.notice" if silent else "m.text"
        base_content: dict[str, Any] = {"msgtype": msgtype, "body": message}
        if silent:
            # Explicitly clear mentions metadata to reduce push noise where supported.
            base_content["m.mentions"] = {}
        if message_format == FORMAT_HTML:
            base_content["format"] = "org.matrix.custom.html"
            base_content["formatted_body"] = message

        if thread_root_event_id:
            thread_reply_event_id = str(reply_to_event_id or thread_root_event_id).strip()
            base_content["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": thread_root_event_id,
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": thread_reply_event_id},
            }
            return base_content

        if reply_to_event_id:
            base_content["m.relates_to"] = {
                "m.in_reply_to": {"event_id": reply_to_event_id}
            }
            return base_content

        if edit_event_id:
            edit_content = {
                "msgtype": msgtype,
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
        silent: bool = False,
        reply_to_event_id: str = "",
        edit_event_id: str = "",
        thread_root_event_id: str = "",
        *,
        queue_on_fail: bool = True,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        content = self._build_message_content(
            message=message,
            message_format=message_format,
            silent=silent,
            reply_to_event_id=reply_to_event_id,
            edit_event_id=edit_event_id,
            thread_root_event_id=thread_root_event_id,
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
                        silent=silent,
                        reply_to_event_id=reply_to_event_id,
                        edit_event_id=edit_event_id,
                        thread_root_event_id=thread_root_event_id,
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
                if queue_on_fail and self._is_transient_error(err):
                    item_id = await self._async_outbox_enqueue(
                        kind="message",
                        payload={
                            "targets": [target],
                            "message": message,
                            "message_format": message_format,
                            "silent": bool(silent),
                            "reply_to_event_id": reply_to_event_id,
                            "edit_event_id": edit_event_id,
                            "thread_root_event_id": thread_root_event_id,
                        },
                        error=str(err),
                    )
                    results.append(
                        {
                            "target": target,
                            "status": "queued",
                            "queue_item_id": item_id,
                            "error": str(err),
                        }
                    )
                else:
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
        *,
        queue_on_fail: bool = True,
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
                if queue_on_fail and self._is_transient_error(err):
                    item_id = await self._async_outbox_enqueue(
                        kind="reaction",
                        payload={
                            "targets": [target],
                            "event_id": event_id,
                            "reaction_key": reaction_key,
                        },
                        error=str(err),
                    )
                    results.append(
                        {
                            "target": target,
                            "status": "queued",
                            "queue_item_id": item_id,
                            "error": str(err),
                        }
                    )
                else:
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

    async def async_redact_event(
        self,
        targets: list[str],
        event_id: str,
        reason: str = "",
        *,
        queue_on_fail: bool = True,
    ) -> dict[str, Any]:
        event_id = str(event_id or "").strip()
        if not event_id:
            raise MatrixChatError("event_id is required")

        reason = str(reason or "").strip()
        results: list[dict[str, Any]] = []
        for target in targets:
            try:
                room_id, target_type = await self._resolve_target_to_room(target)
                encrypted = await self._is_room_encrypted(room_id)
                redaction_event_id = await self._redact_room_event(
                    room_id=room_id,
                    event_id=event_id,
                    reason=reason,
                )
                results.append(
                    {
                        "target": target,
                        "target_type": target_type,
                        "room_id": room_id,
                        "event_id": redaction_event_id,
                        "redacts": event_id,
                        "transport": "direct",
                        "encrypted": encrypted,
                        "status": "sent",
                    }
                )
            except Exception as err:  # noqa: BLE001
                if queue_on_fail and self._is_transient_error(err):
                    item_id = await self._async_outbox_enqueue(
                        kind="redaction",
                        payload={
                            "targets": [target],
                            "event_id": event_id,
                            "reason": reason,
                        },
                        error=str(err),
                    )
                    results.append(
                        {
                            "target": target,
                            "status": "queued",
                            "queue_item_id": item_id,
                            "error": str(err),
                        }
                    )
                else:
                    _LOGGER.exception("Failed redacting event %s in target %s", event_id, target)
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
        thread_root_event_id: str = "",
        *,
        queue_on_fail: bool = True,
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
                        thread_root_event_id=thread_root_event_id,
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
                        self._build_message_content(
                            message=message,
                            message_format="text",
                            silent=False,
                            reply_to_event_id="",
                            edit_event_id="",
                            thread_root_event_id=thread_root_event_id,
                        ),
                    )

                content = {
                    "msgtype": msgtype,
                    "body": media_body,
                    "filename": media_body,
                    "url": content_uri,
                    "info": info,
                }
                if thread_root_event_id:
                    content["m.relates_to"] = {
                        "rel_type": "m.thread",
                        "event_id": thread_root_event_id,
                        "is_falling_back": True,
                        "m.in_reply_to": {"event_id": thread_root_event_id},
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
                if queue_on_fail and self._is_transient_error(err):
                    item_id = await self._async_outbox_enqueue(
                        kind="media",
                        payload={
                            "targets": [target],
                            "file_path": file_path,
                            "message": message,
                            "mime_type": mime_type,
                            "auto_convert": auto_convert,
                            "convert_threshold_mb": convert_threshold_mb,
                            "max_size_mb": max_size_mb,
                            "thread_root_event_id": thread_root_event_id,
                        },
                        error=str(err),
                    )
                    results.append(
                        {
                            "target": target,
                            "status": "queued",
                            "queue_item_id": item_id,
                            "error": str(err),
                        }
                    )
                else:
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
