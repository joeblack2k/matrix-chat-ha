#!/usr/bin/env python3
"""Encrypted Matrix gateway for Home Assistant matrix_chat integration."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from typing import Any

from aiohttp import ClientSession, web
from nio import AsyncClient, AsyncClientConfig
from nio.responses import (
    JoinError,
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


class GatewayError(Exception):
    """Gateway-level error."""


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

        self._client: AsyncClient | None = None
        self._sync_task: asyncio.Task | None = None
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

        sync_response = await self._client.sync(timeout=30000, full_state=True)
        if isinstance(sync_response, SyncError):
            raise GatewayError(
                f"Initial sync failed: {sync_response.status_code} {sync_response.message}"
            )

        self._sync_task = asyncio.create_task(self._sync_loop(), name="matrix-e2ee-sync")
        _LOGGER.info("Matrix gateway ready")

    async def stop(self) -> None:
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
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001
                _LOGGER.exception("Matrix sync loop error: %s", err)
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

    async def send_text(self, room_id: str, message: str, message_format: str) -> str:
        assert self._client is not None

        await self._ensure_joined(room_id)
        content: dict[str, Any] = {"msgtype": "m.text", "body": message}
        if message_format == "html":
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = message

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

    async def send_media(
        self,
        room_id: str,
        file_bytes: bytes,
        msgtype: str,
        body: str,
        caption: str,
        mime_type: str,
        info: dict[str, Any],
    ) -> str:
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
            "info": info,
        }

        if encrypted:
            if not isinstance(maybe_keys, dict):
                raise GatewayError("Encrypted upload missing decryption keys")
            content["file"] = maybe_keys
        else:
            content["url"] = upload_response.content_uri

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

        return send_response.event_id


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
            }
        )

    async def send_text(request: web.Request) -> web.Response:
        payload = await request.json()
        room_id = (payload.get("room_id") or "").strip()
        message = str(payload.get("message") or "")
        message_format = str(payload.get("format") or "text").strip().lower()

        if not room_id:
            raise GatewayError("room_id is required")
        if not message:
            raise GatewayError("message is required")
        if message_format not in {"text", "html"}:
            raise GatewayError("format must be 'text' or 'html'")

        event_id = await gateway.send_text(
            room_id=room_id,
            message=message,
            message_format=message_format,
        )
        return web.json_response({"event_id": event_id})

    async def send_media(request: web.Request) -> web.Response:
        post = await request.post()
        room_id = str(post.get("room_id") or "").strip()
        msgtype = str(post.get("msgtype") or "m.file").strip()
        body = str(post.get("body") or "file").strip()
        caption = str(post.get("caption") or "")
        mime_type = str(post.get("mime_type") or "application/octet-stream").strip()
        info_raw = str(post.get("info") or "{}")

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

        event_id = await gateway.send_media(
            room_id=room_id,
            file_bytes=file_bytes,
            msgtype=msgtype,
            body=body,
            caption=caption,
            mime_type=mime_type,
            info=info,
        )
        return web.json_response({"event_id": event_id})

    app.router.add_get("/health", health)
    app.router.add_post("/send_text", send_text)
    app.router.add_post("/send_media", send_media)

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
