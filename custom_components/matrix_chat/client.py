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

from aiohttp import ClientError, ClientSession
from homeassistant.helpers.storage import Store

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

        self.auto_convert_video = bool(auto_convert_video)
        self.video_convert_threshold_mb = float(video_convert_threshold_mb)
        self.max_upload_mb = float(max_upload_mb)

        self.device_id = ""
        self._server_upload_limit_bytes: int | None = None

        self._store = Store(hass, 1, f"{DOMAIN}_{entry_id}_cache")
        self._dm_rooms: dict[str, str] = {}
        self._encrypted_rooms: dict[str, bool] = {}

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

    async def async_persist_cache(self) -> None:
        await self._store.async_save(
            {
                "dm_rooms": self._dm_rooms,
                "encrypted_rooms": self._encrypted_rooms,
            }
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

    async def _resolve_dm_room(self, target_user: str) -> str:
        if target_user in self._dm_rooms:
            return self._dm_rooms[target_user]

        payload = {
            "is_direct": True,
            "invite": [target_user],
            "preset": "trusted_private_chat",
            "name": f"DM with {target_user}",
        }
        data = await self._request_json(
            "POST",
            "/_matrix/client/v3/createRoom",
            payload=payload,
            expected=(200,),
        )
        room_id = (data or {}).get("room_id", "")
        if not room_id:
            raise MatrixChatError(f"Could not create DM room for {target_user}")

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

    async def _send_encrypted_webhook_message(
        self, room_id: str, message: str, message_format: str
    ) -> str:
        if not self.encrypted_webhook_url:
            raise MatrixChatError(
                f"Room {room_id} is encrypted and no encrypted webhook URL is configured"
            )

        headers = {"Content-Type": "application/json"}
        if self.encrypted_webhook_token:
            headers["Authorization"] = f"Bearer {self.encrypted_webhook_token}"

        payload = {
            "room_id": room_id,
            "sender": self.user_id,
            "message": message,
            "format": message_format,
        }

        async with self._session.post(
            self.encrypted_webhook_url,
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
            raise MatrixChatError(
                f"Encrypted webhook failed HTTP {resp.status}: {text[:500]}"
            )

    async def async_send_message(
        self,
        targets: list[str],
        message: str,
        message_format: str,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []

        for target in targets:
            try:
                room_id, target_type = await self._resolve_target_to_room(target)
                encrypted = await self._is_room_encrypted(room_id)

                if encrypted:
                    event_id = await self._send_encrypted_webhook_message(
                        room_id, message, message_format
                    )
                    results.append(
                        {
                            "target": target,
                            "target_type": target_type,
                            "room_id": room_id,
                            "event_id": event_id,
                            "transport": "encrypted_webhook",
                            "status": "sent",
                        }
                    )
                    continue

                content: dict[str, Any] = {
                    "msgtype": "m.text",
                    "body": message,
                }
                if message_format == FORMAT_HTML:
                    content["format"] = "org.matrix.custom.html"
                    content["formatted_body"] = message

                event_id = await self._send_room_event(room_id, "m.room.message", content)
                results.append(
                    {
                        "target": target,
                        "target_type": target_type,
                        "room_id": room_id,
                        "event_id": event_id,
                        "transport": "direct",
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
        with path.open("rb") as handle:
            async with self._session.post(
                url,
                headers=headers,
                data=handle,
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
        content_uri = await self._upload_media(prepared_path, prepared_mime)

        msgtype = "m.file"
        if prepared_mime.startswith("image/"):
            msgtype = "m.image"
        elif prepared_mime.startswith("video/"):
            msgtype = "m.video"

        body = message or prepared_path.name
        info = {
            "mimetype": prepared_mime,
            "size": prepared_path.stat().st_size,
        }

        results: list[dict[str, Any]] = []
        for target in targets:
            try:
                room_id, target_type = await self._resolve_target_to_room(target)
                encrypted = await self._is_room_encrypted(room_id)

                if encrypted:
                    raise MatrixChatError(
                        f"Room {room_id} is encrypted. Media via encrypted webhook is not supported in v1."
                    )

                content = {
                    "msgtype": msgtype,
                    "body": body,
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
        return {
            "content_uri": content_uri,
            "results": results,
            "success_count": success_count,
            "failure_count": len(results) - success_count,
        }
