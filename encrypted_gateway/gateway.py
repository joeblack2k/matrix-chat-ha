#!/usr/bin/env python3
"""Encrypted Matrix gateway for Home Assistant matrix_chat integration."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import ClientError, ClientSession, web
from nio import AsyncClient, AsyncClientConfig
from nio.exceptions import LocalProtocolError
from nio.responses import (
    JoinError,
    KeysUploadError,
    KeysUploadResponse,
    LoginError,
    LoginResponse,
    RoomGetStateEventError,
    RoomSendError,
    RoomSendResponse,
    SyncError,
    UploadError,
    UploadResponse,
)

_LOGGER = logging.getLogger("matrix_e2ee_gateway")

_MAX_QUEUE_LENGTH = 5000
_DEFAULT_THUMB_MAX_PX = 512
_DEFAULT_THUMB_QUALITY = 75
_DEFAULT_THUMB_MAX_SOURCE_MB = 50


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_token(token: str) -> str:
    out = (token or "").strip()
    if out.lower().startswith("bearer "):
        out = out[7:].strip()
    return out


try:
    from PIL import Image  # type: ignore
except Exception:  # noqa: BLE001
    Image = None  # type: ignore


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        out = int(str(value).strip())
    except Exception:  # noqa: BLE001
        return default
    return max(min_value, min(max_value, out))


def _make_image_thumbnail_jpeg(
    file_bytes: bytes, *, max_px: int, quality: int
) -> tuple[bytes, int, int] | None:
    if Image is None:
        return None
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:  # type: ignore[attr-defined]
            img = img.convert("RGB")
            img.thumbnail((max_px, max_px))
            w, h = img.size
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=quality, optimize=True)
            return out.getvalue(), int(w), int(h)
    except Exception:  # noqa: BLE001
        return None


async def _make_video_thumbnail_jpeg(
    file_bytes: bytes, *, max_px: int, quality: int
) -> tuple[bytes, int, int] | None:
    if Image is None:
        return None
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(file_bytes)
            tmp_path = f.name

        # Extract a single frame and scale it down. Emit JPEG to stdout.
        vf = (
            f"scale='min({max_px},iw)':'min({max_px},ih)':force_original_aspect_ratio=decrease"
        )
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0.5",
            "-i",
            tmp_path,
            "-frames:v",
            "1",
            "-vf",
            vf,
            "-q:v",
            str(max(2, min(31, int(31 - (quality / 100.0) * 29)))),
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await proc.communicate()
        if proc.returncode != 0 or not stdout:
            return None

        with Image.open(io.BytesIO(stdout)) as img:  # type: ignore[attr-defined]
            w, h = img.size
        return stdout, int(w), int(h)
    except Exception:  # noqa: BLE001
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class GatewayError(Exception):
    """Gateway-level error."""


@dataclass
class InboundQueueItem:
    item_id: str
    created_ts: float
    payload: dict[str, Any]
    attempts: int = 0
    next_attempt_ts: float = 0.0
    last_error: str = ""


class InboundDeliveryQueue:
    """Persistent, best-effort delivery queue to Home Assistant webhook."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._items: list[InboundQueueItem] = []
        self.delivered_total = 0
        self.failed_total = 0
        self.last_success_ts: float | None = None
        self.last_error: str = ""
        self._lock = asyncio.Lock()

    def size(self) -> int:
        return len(self._items)

    async def load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            if not raw.strip():
                return
            data = json.loads(raw)
            if not isinstance(data, dict):
                return
            items = data.get("items", [])
            if not isinstance(items, list):
                return

            loaded: list[InboundQueueItem] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                payload = item.get("payload")
                if not isinstance(payload, dict):
                    continue
                item_id = str(item.get("item_id") or "")
                if not item_id:
                    continue
                loaded.append(
                    InboundQueueItem(
                        item_id=item_id,
                        created_ts=float(item.get("created_ts") or 0.0),
                        payload=payload,
                        attempts=int(item.get("attempts") or 0),
                        next_attempt_ts=float(item.get("next_attempt_ts") or 0.0),
                        last_error=str(item.get("last_error") or ""),
                    )
                )

            self._items = loaded[-_MAX_QUEUE_LENGTH:]
            self.delivered_total = int(data.get("delivered_total") or 0)
            self.failed_total = int(data.get("failed_total") or 0)
            last_success = data.get("last_success_ts")
            self.last_success_ts = float(last_success) if last_success else None
            self.last_error = str(data.get("last_error") or "")
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Could not load inbound queue: %s", err)

    async def _persist(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        data = {
            "items": [
                {
                    "item_id": i.item_id,
                    "created_ts": i.created_ts,
                    "payload": i.payload,
                    "attempts": i.attempts,
                    "next_attempt_ts": i.next_attempt_ts,
                    "last_error": i.last_error,
                }
                for i in self._items
            ],
            "delivered_total": self.delivered_total,
            "failed_total": self.failed_total,
            "last_success_ts": self.last_success_ts,
            "last_error": self.last_error,
        }
        tmp.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
        tmp.replace(self._path)

    async def enqueue(self, payload: dict[str, Any]) -> str:
        now = time.time()
        item_id = f"in-{int(now * 1000)}-{os.urandom(4).hex()}"
        item = InboundQueueItem(item_id=item_id, created_ts=now, payload=payload)
        async with self._lock:
            self._items.append(item)
            self._items = self._items[-_MAX_QUEUE_LENGTH:]
            await self._persist()
        return item_id

    def next_due(self, now: float) -> InboundQueueItem | None:
        for item in self._items:
            if item.next_attempt_ts <= now:
                return item
        return None

    async def mark_success(self, item_id: str) -> None:
        async with self._lock:
            self._items = [i for i in self._items if i.item_id != item_id]
            self.delivered_total += 1
            self.last_success_ts = time.time()
            self.last_error = ""
            await self._persist()

    async def mark_failure(self, item_id: str, err: str) -> None:
        async with self._lock:
            for item in self._items:
                if item.item_id == item_id:
                    item.attempts += 1
                    item.last_error = err[:500]
                    delay = min(300, max(1, 2 ** min(item.attempts, 8)))
                    item.next_attempt_ts = time.time() + delay
                    break
            self.failed_total += 1
            self.last_error = err[:500]
            await self._persist()


class MatrixE2EEGateway:
    """Matrix E2EE sending gateway."""

    def __init__(self) -> None:
        self.homeserver = _env("MATRIX_HOMESERVER")
        self.user_id = _env("MATRIX_USER_ID")
        self.password = _env("MATRIX_PASSWORD")
        self.access_token = _normalize_token(_env("MATRIX_ACCESS_TOKEN"))
        self.device_id = _env("MATRIX_DEVICE_ID")
        self.device_name = _env("MATRIX_DEVICE_NAME", "HA Matrix E2EE Gateway")
        self.store_path = _env("MATRIX_STORE_PATH", "/data/store")
        self.api_token = _normalize_token(_env("MATRIX_GATEWAY_TOKEN"))
        self.verify_ssl = _env_bool("MATRIX_VERIFY_SSL", True)
        self.listen_host = _env("MATRIX_GATEWAY_HOST", "0.0.0.0")
        self.listen_port = int(_env("MATRIX_GATEWAY_PORT", "8080"))
        self.ignore_unverified_devices = _env_bool(
            "MATRIX_IGNORE_UNVERIFIED_DEVICES", True
        )
        self.inbound_webhook_url = _env("MATRIX_INBOUND_WEBHOOK_URL")
        self.inbound_shared_secret = _env("MATRIX_INBOUND_SHARED_SECRET")
        self.debug_endpoints = _env_bool("MATRIX_DEBUG_ENDPOINTS", False)
        self.data_dir = Path(_env("MATRIX_DATA_DIR", "/data"))
        self._queue = InboundDeliveryQueue(self.data_dir / "inbound_queue.json")

        self._client: AsyncClient | None = None
        self._sync_task: asyncio.Task | None = None
        self._deliver_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()

    async def _whoami_device_id(self) -> str:
        if not self.access_token:
            raise GatewayError("MATRIX_ACCESS_TOKEN missing")

        headers = {"Authorization": f"Bearer {self.access_token}"}
        url = f"{self.homeserver.rstrip('/')}/_matrix/client/v3/account/whoami"

        async with ClientSession() as session:
            async with session.get(url, headers=headers, ssl=self.verify_ssl) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise GatewayError(
                        f"whoami failed HTTP {resp.status}: {text[:300]}"
                    )
                data = json.loads(text)

        whoami_user = data.get("user_id")
        if whoami_user and whoami_user != self.user_id:
            raise GatewayError(
                f"whoami user mismatch: configured={self.user_id}, token={whoami_user}"
            )

        device_id = (data.get("device_id") or "").strip()
        if device_id:
            return device_id

        # Some deployments omit device_id in whoami; fallback to /devices.
        devices_url = f"{self.homeserver.rstrip('/')}/_matrix/client/v3/devices"
        async with ClientSession() as session:
            async with session.get(devices_url, headers=headers, ssl=self.verify_ssl) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise GatewayError(
                        f"devices lookup failed HTTP {resp.status}: {text[:300]}"
                    )
                devices_data = json.loads(text)

        devices = devices_data.get("devices") or []
        if not isinstance(devices, list) or not devices:
            raise GatewayError("Could not determine device_id from token")

        # Prefer the most recently seen device if available.
        def _last_seen(item: dict[str, Any]) -> int:
            value = item.get("last_seen_ts")
            return int(value) if isinstance(value, int) else 0

        best = max((d for d in devices if isinstance(d, dict)), key=_last_seen, default={})
        fallback_device_id = str(best.get("device_id") or "").strip()
        if not fallback_device_id:
            raise GatewayError("Token devices did not include a valid device_id")
        return fallback_device_id

    async def _ensure_device_keys_uploaded(self) -> None:
        assert self._client is not None

        for attempt in range(1, 11):
            try:
                response = await self._client.keys_upload()
            except LocalProtocolError as err:
                # matrix-nio raises this when no key upload is required.
                if "No key upload needed" in str(err):
                    _LOGGER.info("Matrix device keys already up to date")
                    return
                raise GatewayError(f"Matrix keys_upload local protocol error: {err}") from err

            if isinstance(response, KeysUploadResponse):
                _LOGGER.info(
                    "Matrix device keys uploaded/verified (curve=%s signed_curve=%s)",
                    response.curve25519_count,
                    response.signed_curve25519_count,
                )
                return

            if isinstance(response, KeysUploadError):
                status = str(response.status_code)
                if status in {"429", "M_LIMIT_EXCEEDED"}:
                    retry_after_ms = getattr(response, "retry_after_ms", 0) or 0
                    retry_seconds = (
                        max(1, int(retry_after_ms / 1000))
                        if retry_after_ms
                        else min(60, attempt * 3)
                    )
                    _LOGGER.warning(
                        "Matrix keys_upload rate-limited (attempt %s/10), retry in %ss",
                        attempt,
                        retry_seconds,
                    )
                    await asyncio.sleep(retry_seconds)
                    continue

                raise GatewayError(
                    f"Matrix keys_upload failed: {response.status_code} {response.message}"
                )

            raise GatewayError("Matrix keys_upload failed with unknown response")

        raise GatewayError("Matrix keys_upload failed after retries")

    async def start(self) -> None:
        if not self.homeserver:
            raise GatewayError("MATRIX_HOMESERVER is required")
        if not self.user_id:
            raise GatewayError("MATRIX_USER_ID is required")
        if not self.api_token:
            raise GatewayError("MATRIX_GATEWAY_TOKEN is required")
        if not self.password and not self.access_token:
            raise GatewayError("Provide MATRIX_PASSWORD or MATRIX_ACCESS_TOKEN")

        if self.access_token and not self.device_id:
            self.device_id = await self._whoami_device_id()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        await self._queue.load()

        config = AsyncClientConfig(
            max_limit_exceeded=0,
            max_timeouts=0,
            store_sync_tokens=True,
            encryption_enabled=True,
        )

        self._client = AsyncClient(
            homeserver=self.homeserver,
            user=self.user_id,
            device_id=self.device_id or "",
            store_path=self.store_path,
            config=config,
            ssl=self.verify_ssl,
        )

        if self.access_token:
            self._client.restore_login(
                user_id=self.user_id,
                device_id=self.device_id,
                access_token=self.access_token,
            )
            self._client.load_store()
            _LOGGER.info("Restored Matrix session for %s (%s)", self.user_id, self.device_id)
        else:
            login_response: LoginResponse | LoginError | Any = None
            for attempt in range(1, 21):
                login_response = await self._client.login(
                    password=self.password,
                    device_name=self.device_name,
                )
                if isinstance(login_response, LoginResponse):
                    break

                if isinstance(login_response, LoginError):
                    status = str(login_response.status_code)
                    if status in {"429", "M_LIMIT_EXCEEDED"}:
                        retry_after_ms = getattr(login_response, "retry_after_ms", 0) or 0
                        retry_seconds = max(1, int(retry_after_ms / 1000)) if retry_after_ms else min(
                            60, attempt * 3
                        )
                        _LOGGER.warning(
                            "Matrix login rate-limited (attempt %s/20), retry in %ss",
                            attempt,
                            retry_seconds,
                        )
                        await asyncio.sleep(retry_seconds)
                        continue
                break

            if not isinstance(login_response, LoginResponse):
                if isinstance(login_response, LoginError):
                    raise GatewayError(
                        f"Matrix login failed: {login_response.status_code} {login_response.message}"
                    )
                raise GatewayError("Matrix login failed with unknown response")
            self.access_token = login_response.access_token
            self.device_id = login_response.device_id
            _LOGGER.info("Logged in Matrix gateway for %s (%s)", self.user_id, self.device_id)

        await self._ensure_device_keys_uploaded()

        sync_response = await self._client.sync(timeout=30000, full_state=True)
        if isinstance(sync_response, SyncError):
            raise GatewayError(
                f"Initial sync failed: {sync_response.status_code} {sync_response.message}"
            )

        self._sync_task = asyncio.create_task(self._sync_loop(), name="matrix-e2ee-sync")
        self._deliver_task = asyncio.create_task(
            self._deliver_loop(), name="matrix-inbound-deliver"
        )
        _LOGGER.info("Matrix gateway ready")

    async def stop(self) -> None:
        if self._deliver_task:
            self._deliver_task.cancel()
            try:
                await self._deliver_task
            except asyncio.CancelledError:
                pass
            self._deliver_task = None

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        if self._client:
            await self._client.close()
            self._client = None

    async def _sync_loop(self) -> None:
        assert self._client is not None
        while True:
            try:
                response = await self._client.sync(timeout=30000, full_state=False)
                if isinstance(response, SyncError):
                    _LOGGER.warning(
                        "Sync error: %s %s",
                        response.status_code,
                        response.message,
                    )
                    await asyncio.sleep(2)
                    continue

                await self._process_inbound_sync(response)
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001
                _LOGGER.exception("Matrix sync loop error: %s", err)
                await asyncio.sleep(2)

    async def _process_inbound_sync(self, response: Any) -> None:
        """Extract inbound room events and enqueue for HA webhook delivery."""
        if not self.inbound_webhook_url:
            return

        rooms = getattr(getattr(response, "rooms", None), "join", None) or {}
        for room_id, room_data in rooms.items():
            timeline = getattr(room_data, "timeline", None)
            events = getattr(timeline, "events", None) or []
            for ev in events:
                src = getattr(ev, "source", None) or {}
                if not isinstance(src, dict):
                    continue

                sender = str(src.get("sender") or "")
                if sender and sender == self.user_id:
                    continue

                event_id = str(src.get("event_id") or getattr(ev, "event_id", "") or "")
                ev_type = str(src.get("type") or "")
                content = src.get("content") if isinstance(src.get("content"), dict) else {}

                if ev_type not in {"m.room.message", "m.reaction"}:
                    continue

                payload: dict[str, Any] = {
                    "room_id": room_id,
                    "event_id": event_id,
                    "sender": sender,
                    "type": ev_type,
                    "content": content,
                    "msgtype": str(content.get("msgtype") or ""),
                    "body": content.get("body") if isinstance(content, dict) else None,
                    "origin_server_ts": src.get("origin_server_ts"),
                }
                await self._queue.enqueue(payload)

    async def _deliver_loop(self) -> None:
        """Deliver queued inbound events to Home Assistant webhook with retries."""
        if not self.inbound_webhook_url:
            while True:
                await asyncio.sleep(10)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.inbound_shared_secret:
            headers["X-Matrix-Chat-Secret"] = self.inbound_shared_secret

        while True:
            try:
                now = time.time()
                item = self._queue.next_due(now)
                if item is None:
                    await asyncio.sleep(1)
                    continue

                try:
                    async with ClientSession() as session:
                        async with session.post(
                            self.inbound_webhook_url,
                            headers=headers,
                            json=item.payload,
                            ssl=self.verify_ssl,
                            timeout=10,
                        ) as resp:
                            text = await resp.text()
                            if 200 <= resp.status < 300:
                                await self._queue.mark_success(item.item_id)
                            else:
                                await self._queue.mark_failure(
                                    item.item_id, f"HTTP {resp.status}: {text[:400]}"
                                )
                except (ClientError, asyncio.TimeoutError) as err:
                    await self._queue.mark_failure(
                        item.item_id, f"{type(err).__name__}: {err}"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001
                _LOGGER.exception("Inbound delivery loop error: %s", err)
                await asyncio.sleep(2)

    async def _ensure_joined(self, room_id: str) -> None:
        assert self._client is not None
        if room_id in self._client.rooms:
            return

        join_response = await self._client.join(room_id)
        if isinstance(join_response, JoinError):
            raise GatewayError(
                f"Join room failed {room_id}: {join_response.status_code} {join_response.message}"
            )

    async def _is_room_encrypted(self, room_id: str) -> bool:
        assert self._client is not None
        room = self._client.rooms.get(room_id)
        if room is not None:
            return bool(room.encrypted)

        state_response = await self._client.room_get_state_event(
            room_id=room_id,
            event_type="m.room.encryption",
            state_key="",
        )
        return not isinstance(state_response, RoomGetStateEventError)

    def _build_message_content(
        self,
        message: str,
        message_format: str,
        reply_to_event_id: str,
        edit_event_id: str,
    ) -> dict[str, Any]:
        if reply_to_event_id and edit_event_id:
            raise GatewayError("reply_to_event_id and edit_event_id are mutually exclusive")

        content: dict[str, Any] = {"msgtype": "m.text", "body": message}
        if message_format == "html":
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = message

        if reply_to_event_id:
            content["m.relates_to"] = {
                "m.in_reply_to": {"event_id": reply_to_event_id}
            }
            return content

        if edit_event_id:
            edit_content: dict[str, Any] = {
                "msgtype": "m.text",
                "body": f"* {message}",
                "m.new_content": dict(content),
                "m.relates_to": {
                    "rel_type": "m.replace",
                    "event_id": edit_event_id,
                },
            }
            if message_format == "html":
                edit_content["format"] = "org.matrix.custom.html"
                edit_content["formatted_body"] = f"* {message}"
            return edit_content

        return content

    async def send_text(
        self,
        room_id: str,
        message: str,
        message_format: str,
        reply_to_event_id: str = "",
        edit_event_id: str = "",
    ) -> str:
        assert self._client is not None

        await self._ensure_joined(room_id)
        content = self._build_message_content(
            message=message,
            message_format=message_format,
            reply_to_event_id=reply_to_event_id,
            edit_event_id=edit_event_id,
        )

        async with self._send_lock:
            response = await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
                ignore_unverified_devices=self.ignore_unverified_devices,
            )

        if not isinstance(response, RoomSendResponse):
            if isinstance(response, RoomSendError):
                raise GatewayError(
                    f"Send text failed {response.status_code}: {response.message}"
                )
            raise GatewayError("Send text failed with unknown response")
        return response.event_id

    async def send_reaction(self, room_id: str, event_id: str, reaction_key: str) -> str:
        assert self._client is not None
        await self._ensure_joined(room_id)

        content: dict[str, Any] = {
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": event_id,
                "key": reaction_key,
            }
        }

        async with self._send_lock:
            response = await self._client.room_send(
                room_id=room_id,
                message_type="m.reaction",
                content=content,
                ignore_unverified_devices=self.ignore_unverified_devices,
            )

        if not isinstance(response, RoomSendResponse):
            if isinstance(response, RoomSendError):
                raise GatewayError(
                    f"Send reaction failed {response.status_code}: {response.message}"
                )
            raise GatewayError("Send reaction failed with unknown response")
        return response.event_id

    async def send_media(
        self,
        room_id: str,
        file_bytes: bytes,
        msgtype: str,
        body: str,
        caption: str,
        mime_type: str,
        info: dict[str, Any],
        *,
        thumbnails_enabled: bool = True,
        thumb_max_px: int = _DEFAULT_THUMB_MAX_PX,
        thumb_quality: int = _DEFAULT_THUMB_QUALITY,
        thumb_max_source_mb: int = _DEFAULT_THUMB_MAX_SOURCE_MB,
    ) -> tuple[str, dict[str, Any] | None]:
        assert self._client is not None

        await self._ensure_joined(room_id)
        encrypted = await self._is_room_encrypted(room_id)

        async with self._send_lock:
            upload_response, maybe_keys = await self._client.upload(
                io.BytesIO(file_bytes),
                content_type=mime_type,
                filename=body,
                filesize=len(file_bytes),
                encrypt=encrypted,
            )

        if not isinstance(upload_response, UploadResponse):
            if isinstance(upload_response, UploadError):
                raise GatewayError(
                    f"Upload failed {upload_response.status_code}: {upload_response.message}"
                )
            raise GatewayError("Upload failed with unknown response")

        content: dict[str, Any] = {
            "msgtype": msgtype,
            "body": body,
            "filename": body,
            "info": dict(info or {}),
        }

        debug: dict[str, Any] | None = None
        thumb_added = False
        if encrypted:
            if not isinstance(maybe_keys, dict):
                raise GatewayError("Encrypted upload missing decryption keys")

            # Force a plain dict with a valid MXC URL in file.url for strict clients
            # (Matrix clients may reject encrypted media events when file.url is missing).
            file_content = dict(maybe_keys)
            encrypted_url = file_content.get("url") or upload_response.content_uri
            if not encrypted_url:
                raise GatewayError("Encrypted upload missing content_uri/file.url")

            encrypted_url_str = str(encrypted_url).strip()
            if not encrypted_url_str:
                raise GatewayError("Encrypted upload returned empty file.url")

            file_content["url"] = encrypted_url_str
            # Some clients (notably Element X) also expect mimetype/size on the file object.
            file_content.setdefault("mimetype", mime_type)
            file_content.setdefault("size", len(file_bytes))
            content["file"] = file_content
            # Spec-compliant encrypted media: use only `file` (omit top-level `url`).
            # Some clients will treat the presence of `url` as unencrypted media and
            # then fail to decrypt/play the attachment.

            if self.debug_endpoints:
                debug = {
                    "encrypted": True,
                    "msgtype": msgtype,
                    "has_url": bool(content.get("url")),
                    "has_file": True,
                    "file_keys": sorted(str(k) for k in file_content.keys()),
                }
        else:
            content["url"] = upload_response.content_uri
            if self.debug_endpoints:
                debug = {
                    "encrypted": False,
                    "msgtype": msgtype,
                    "has_url": bool(content.get("url")),
                    "has_file": False,
                    "file_keys": [],
                }

        # Optional media thumbnail for richer previews (encrypted rooms: encrypted thumbnail_file).
        if (
            thumbnails_enabled
            and msgtype in {"m.image", "m.video"}
            and file_bytes
            and len(file_bytes) <= int(thumb_max_source_mb) * 1024 * 1024
        ):
            try:
                thumb: tuple[bytes, int, int] | None
                if msgtype == "m.image":
                    thumb = _make_image_thumbnail_jpeg(
                        file_bytes, max_px=int(thumb_max_px), quality=int(thumb_quality)
                    )
                else:
                    thumb = await _make_video_thumbnail_jpeg(
                        file_bytes, max_px=int(thumb_max_px), quality=int(thumb_quality)
                    )

                if thumb:
                    thumb_bytes, tw, th = thumb
                    async with self._send_lock:
                        thumb_upload, thumb_keys = await self._client.upload(
                            io.BytesIO(thumb_bytes),
                            content_type="image/jpeg",
                            filename="thumbnail.jpg",
                            filesize=len(thumb_bytes),
                            encrypt=encrypted,
                        )

                    if not isinstance(thumb_upload, UploadResponse):
                        raise GatewayError("Thumbnail upload failed")

                    thumb_mxc = str(thumb_upload.content_uri or "").strip()
                    if not thumb_mxc:
                        raise GatewayError("Thumbnail upload missing content_uri")

                    info_obj = content.setdefault("info", {})
                    if isinstance(info_obj, dict):
                        info_obj["thumbnail_info"] = {
                            "mimetype": "image/jpeg",
                            "size": len(thumb_bytes),
                            "w": tw,
                            "h": th,
                        }
                        if not encrypted:
                            # For unencrypted rooms, include thumbnail_url so the server/client can fetch it directly.
                            info_obj["thumbnail_url"] = thumb_mxc

                        if encrypted:
                            if not isinstance(thumb_keys, dict):
                                raise GatewayError("Encrypted thumbnail upload missing keys")
                            thumb_file = dict(thumb_keys)
                            thumb_file["url"] = thumb_mxc
                            thumb_file.setdefault("mimetype", "image/jpeg")
                            thumb_file.setdefault("size", len(thumb_bytes))
                            info_obj["thumbnail_file"] = thumb_file

                    thumb_added = True
            except Exception as err:  # noqa: BLE001
                _LOGGER.debug("Thumbnail generation/upload failed: %s", err)

        if caption:
            await self.send_text(room_id=room_id, message=caption, message_format="text")

        async with self._send_lock:
            send_response = await self._client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
                ignore_unverified_devices=self.ignore_unverified_devices,
            )

        if not isinstance(send_response, RoomSendResponse):
            if isinstance(send_response, RoomSendError):
                raise GatewayError(
                    f"Send media failed {send_response.status_code}: {send_response.message}"
                )
            raise GatewayError("Send media failed with unknown response")

        _LOGGER.info(
            "Media sent: room=%s msgtype=%s encrypted=%s thumbnail=%s",
            room_id,
            msgtype,
            encrypted,
            thumb_added,
        )
        if debug is None:
            debug = {"thumbnail_added": thumb_added}
        else:
            debug["thumbnail_added"] = thumb_added

        return send_response.event_id, debug


@web.middleware
async def json_error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except GatewayError as err:
        _LOGGER.warning("Gateway request error: %s", err)
        return web.json_response({"error": str(err)}, status=400)
    except Exception as err:  # noqa: BLE001
        _LOGGER.exception("Unhandled gateway error")
        return web.json_response({"error": f"internal_error: {err}"}, status=500)


def auth_middleware(expected_token: str):
    @web.middleware
    async def _auth(request: web.Request, handler):
        if request.path == "/health":
            return await handler(request)

        auth = request.headers.get("Authorization", "")
        token = auth
        if auth.lower().startswith("bearer "):
            token = auth[7:]
        token = token.strip()

        if not token or token != expected_token:
            return web.json_response({"error": "unauthorized"}, status=401)

        return await handler(request)

    return _auth


async def create_app() -> web.Application:
    gateway = MatrixE2EEGateway()
    app = web.Application(
        middlewares=[auth_middleware(gateway.api_token), json_error_middleware],
        client_max_size=512 * 1024**2,
    )
    app["gateway"] = gateway

    async def on_startup(_app: web.Application) -> None:
        await gateway.start()

    async def on_cleanup(_app: web.Application) -> None:
        await gateway.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    async def health(_request: web.Request) -> web.Response:
        client = gateway._client
        rooms = len(client.rooms) if client else 0
        return web.json_response(
            {
                "status": "ok",
                "user_id": gateway.user_id,
                "device_id": gateway.device_id,
                "rooms_known": rooms,
                "inbound_enabled": bool(gateway.inbound_webhook_url),
                "inbound_queue_size": gateway._queue.size(),
                "inbound_delivered_total": gateway._queue.delivered_total,
                "inbound_failed_total": gateway._queue.failed_total,
                "inbound_last_success_ts": gateway._queue.last_success_ts,
                "inbound_last_error": gateway._queue.last_error,
            }
        )

    async def stats(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "inbound_enabled": bool(gateway.inbound_webhook_url),
                "inbound_queue_size": gateway._queue.size(),
                "inbound_delivered_total": gateway._queue.delivered_total,
                "inbound_failed_total": gateway._queue.failed_total,
                "inbound_last_success_ts": gateway._queue.last_success_ts,
                "inbound_last_error": gateway._queue.last_error,
            }
        )

    async def simulate_inbound(request: web.Request) -> web.Response:
        if not gateway.debug_endpoints:
            return web.json_response({"error": "disabled"}, status=404)
        payload = await request.json()
        if not isinstance(payload, dict):
            raise GatewayError("payload must be object")
        item_id = await gateway._queue.enqueue(payload)
        return web.json_response({"status": "queued", "item_id": item_id})

    async def send_text(request: web.Request) -> web.Response:
        payload = await request.json()
        room_id = (payload.get("room_id") or "").strip()
        message = str(payload.get("message") or "")
        message_format = str(payload.get("format") or "text").strip().lower()
        reply_to_event_id = str(payload.get("reply_to_event_id") or "").strip()
        edit_event_id = str(payload.get("edit_event_id") or "").strip()

        if not room_id:
            raise GatewayError("room_id is required")
        if not message:
            raise GatewayError("message is required")
        if message_format not in {"text", "html"}:
            raise GatewayError("format must be 'text' or 'html'")
        if reply_to_event_id and edit_event_id:
            raise GatewayError("reply_to_event_id and edit_event_id are mutually exclusive")

        # Low-noise debug aid: when HA inbound commands are enabled, the integration
        # replies with "Command ...". Log those replies to help troubleshoot allowlists
        # without dumping all outbound bot traffic.
        if message.startswith("Command "):
            _LOGGER.warning("HA inbound command reply: room=%s msg=%s", room_id, message[:300])

        event_id = await gateway.send_text(
            room_id=room_id,
            message=message,
            message_format=message_format,
            reply_to_event_id=reply_to_event_id,
            edit_event_id=edit_event_id,
        )
        return web.json_response({"event_id": event_id})

    async def send_reaction(request: web.Request) -> web.Response:
        payload = await request.json()
        room_id = str(payload.get("room_id") or "").strip()
        event_id = str(payload.get("event_id") or "").strip()
        reaction_key = str(payload.get("reaction_key") or "").strip()

        if not room_id:
            raise GatewayError("room_id is required")
        if not event_id:
            raise GatewayError("event_id is required")
        if not reaction_key:
            raise GatewayError("reaction_key is required")

        reaction_event_id = await gateway.send_reaction(
            room_id=room_id,
            event_id=event_id,
            reaction_key=reaction_key,
        )
        return web.json_response({"event_id": reaction_event_id})

    async def send_media(request: web.Request) -> web.Response:
        post = await request.post()
        room_id = str(post.get("room_id") or "").strip()
        msgtype = str(post.get("msgtype") or "m.file").strip()
        body = str(post.get("body") or "file").strip()
        caption = str(post.get("caption") or "")
        mime_type = str(post.get("mime_type") or "application/octet-stream").strip()
        info_raw = str(post.get("info") or "{}")
        thumbnails_enabled = _parse_bool(post.get("thumbnails_enabled"), True)
        thumb_max_px = _clamp_int(
            post.get("thumb_max_px"), _DEFAULT_THUMB_MAX_PX, 64, 2048
        )
        thumb_quality = _clamp_int(post.get("thumb_quality"), _DEFAULT_THUMB_QUALITY, 30, 95)
        thumb_max_source_mb = _clamp_int(
            post.get("thumb_max_source_mb"), _DEFAULT_THUMB_MAX_SOURCE_MB, 1, 512
        )

        if not room_id:
            raise GatewayError("room_id is required")

        file_field = post.get("file")
        if file_field is None or not hasattr(file_field, "file"):
            raise GatewayError("file is required")

        try:
            info = json.loads(info_raw)
            if not isinstance(info, dict):
                info = {}
        except json.JSONDecodeError:
            info = {}

        file_bytes = file_field.file.read()
        if not file_bytes:
            raise GatewayError("file is empty")

        event_id, debug = await gateway.send_media(
            room_id=room_id,
            file_bytes=file_bytes,
            msgtype=msgtype,
            body=body,
            caption=caption,
            mime_type=mime_type,
            info=info,
            thumbnails_enabled=thumbnails_enabled,
            thumb_max_px=thumb_max_px,
            thumb_quality=thumb_quality,
            thumb_max_source_mb=thumb_max_source_mb,
        )
        response: dict[str, Any] = {"event_id": event_id}
        if debug:
            response["debug"] = debug
        return web.json_response(response)

    app.router.add_get("/health", health)
    app.router.add_get("/stats", stats)
    app.router.add_post("/send_text", send_text)
    app.router.add_post("/send_media", send_media)
    app.router.add_post("/send_reaction", send_reaction)
    app.router.add_post("/simulate_inbound", simulate_inbound)

    return app


def main() -> None:
    log_level = _env("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    host = _env("MATRIX_GATEWAY_HOST", "0.0.0.0")
    port = int(_env("MATRIX_GATEWAY_PORT", "8080"))
    app = asyncio.run(create_app())
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()
