"""Matrix Chat custom integration."""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from typing import Any

import voluptuous as vol
from aiohttp import web

from homeassistant.components import webhook
from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import CONF_PASSWORD
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.exceptions import ConfigEntryAuthFailed, HomeAssistantError
from homeassistant.helpers import aiohttp_client, config_validation as cv
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers.update_coordinator import ConfigEntryNotReady
from homeassistant.util import dt as dt_util

from .client import (
    MatrixChatAuthError,
    MatrixChatClient,
    MatrixChatConnectionError,
    _normalize_token,
)
from .coordinator import MatrixChatGatewayCoordinator
from .const import (
    ATTR_AUTO_CONVERT,
    ATTR_CONVERT_THRESHOLD_MB,
    ATTR_EDIT_EVENT_ID,
    ATTR_ENTRY_ID,
    ATTR_EVENT_ID,
    ATTR_FILE_PATH,
    ATTR_FORMAT,
    ATTR_THREAD_ROOT_EVENT_ID,
    ATTR_SILENT,
    ATTR_REASON,
    ATTR_INBOUND_ENABLED,
    ATTR_INBOUND_EVENT_TYPE,
    ATTR_INBOUND_WEBHOOK_ID,
    ATTR_INBOUND_WEBHOOK_PATH,
    ATTR_MAX_SIZE_MB,
    ATTR_MESSAGE,
    ATTR_MIME_TYPE,
    ATTR_OUTBOX_LAST_ERROR,
    ATTR_OUTBOX_OLDEST_TS,
    ATTR_OUTBOX_SIZE,
    ATTR_ROOM_ID,
    ATTR_ROOM_OR_ALIAS,
    ATTR_REACTION_KEY,
    ATTR_REPLY_TO_EVENT_ID,
    ATTR_TARGET,
    ATTR_TARGETS,
    ATTR_USER_ID,
    ATTR_INCLUDE_MEMBERS,
    ATTR_LIMIT,
    ATTR_DRY_RUN,
    CONF_ACCESS_TOKEN,
    CONF_AUTO_CONVERT_VIDEO,
    CONF_COMMANDS_ALLOWED_ROOMS,
    CONF_COMMANDS_ALLOWED_SENDERS,
    CONF_COMMANDS_ALLOWED_SERVICES,
    CONF_COMMANDS_ENABLED,
    CONF_DM_ENCRYPTED,
    CONF_ENCRYPTED_WEBHOOK_TOKEN,
    CONF_ENCRYPTED_WEBHOOK_URL,
    CONF_HOMESERVER,
    CONF_INBOUND_ENABLED,
    CONF_INBOUND_SHARED_SECRET,
    CONF_MAX_UPLOAD_MB,
    CONF_USER_ID,
    CONF_VERIFY_SSL,
    CONF_VIDEO_CONVERT_THRESHOLD_MB,
    DEFAULT_AUTO_CONVERT_VIDEO,
    DEFAULT_COMMANDS_ENABLED,
    DEFAULT_DM_ENCRYPTED,
    DEFAULT_INBOUND_ENABLED,
    DEFAULT_MAX_UPLOAD_MB,
    DEFAULT_VERIFY_SSL,
    DEFAULT_VIDEO_CONVERT_THRESHOLD_MB,
    DOMAIN,
    EVENT_INBOUND_MESSAGE,
    FORMAT_HTML,
    FORMAT_TEXT,
    SERVICE_GET_INBOUND_CONFIG,
    SERVICE_FLUSH_OUTBOX,
    SERVICE_GET_OUTBOX_STATS,
    SERVICE_RESOLVE_TARGET,
    SERVICE_LIST_ROOMS,
    SERVICE_JOIN_ROOM,
    SERVICE_INVITE_USER,
    SERVICE_ENSURE_DM,
    SERVICE_ENSURE_ROOM_ENCRYPTED,
    SERVICE_SEND_MEDIA,
    SERVICE_SEND_MESSAGE,
    SERVICE_SEND_REACTION,
    SERVICE_REDACT_EVENT,
)

_LOGGER = logging.getLogger(__name__)
PLATFORMS: list[str] = ["sensor", "binary_sensor"]

_COMMAND_PREFIX = "!ha"
# domain.service (literal dot). Note: raw string should use '\.' not '\\.'.
_SERVICE_RE = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
_NOTIFY_SERVICE_NAME = "matrix_chat"

_NOTIFY_MESSAGE = "message"
_NOTIFY_TITLE = "title"
_NOTIFY_TARGET = "target"
_NOTIFY_DATA = "data"

_NOTIFY_DATA_FILE_PATH = "file_path"
_NOTIFY_DATA_FILE_PATHS = "file_paths"
_NOTIFY_DATA_MIME_TYPE = "mime_type"
_NOTIFY_DATA_AUTO_CONVERT = "auto_convert"
_NOTIFY_DATA_CONVERT_THRESHOLD_MB = "convert_threshold_mb"
_NOTIFY_DATA_MAX_SIZE_MB = "max_size_mb"
_NOTIFY_DATA_SILENT = "silent"
_NOTIFY_DATA_THREAD_ROOT_EVENT_ID = "thread_root_event_id"

MATRIX_CHAT_CONFIG_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_HOMESERVER): cv.url,
        vol.Required(CONF_USER_ID): cv.string,
        vol.Optional(CONF_PASSWORD, default=""): cv.string,
        vol.Optional(CONF_ACCESS_TOKEN, default=""): cv.string,
        vol.Optional(CONF_VERIFY_SSL, default=DEFAULT_VERIFY_SSL): cv.boolean,
        vol.Optional(CONF_ENCRYPTED_WEBHOOK_URL, default=""): cv.string,
        vol.Optional(CONF_ENCRYPTED_WEBHOOK_TOKEN, default=""): cv.string,
        vol.Optional(CONF_DM_ENCRYPTED, default=DEFAULT_DM_ENCRYPTED): cv.boolean,
        vol.Optional(CONF_AUTO_CONVERT_VIDEO, default=DEFAULT_AUTO_CONVERT_VIDEO): cv.boolean,
        vol.Optional(
            CONF_VIDEO_CONVERT_THRESHOLD_MB, default=DEFAULT_VIDEO_CONVERT_THRESHOLD_MB
        ): vol.Coerce(float),
        vol.Optional(CONF_MAX_UPLOAD_MB, default=DEFAULT_MAX_UPLOAD_MB): vol.Coerce(float),
        vol.Optional(CONF_INBOUND_ENABLED, default=DEFAULT_INBOUND_ENABLED): cv.boolean,
        vol.Optional(CONF_INBOUND_SHARED_SECRET, default=""): cv.string,
        vol.Optional(CONF_COMMANDS_ENABLED, default=DEFAULT_COMMANDS_ENABLED): cv.boolean,
        vol.Optional(CONF_COMMANDS_ALLOWED_SENDERS, default=""): cv.string,
        vol.Optional(CONF_COMMANDS_ALLOWED_ROOMS, default=""): cv.string,
        vol.Optional(CONF_COMMANDS_ALLOWED_SERVICES, default=""): cv.string,
    }
)

NOTIFY_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Required(_NOTIFY_MESSAGE): cv.string,
        # Keep HA notify signature compatibility; title is optional and ignored by Matrix.
        vol.Optional(_NOTIFY_TITLE, default=""): cv.string,
        # HA notify uses "target" (list); we accept string or list of strings.
        vol.Optional(_NOTIFY_TARGET): vol.Any(cv.string, [cv.string]),
        # Optional advanced parameters for attachments.
        vol.Optional(_NOTIFY_DATA, default={}): dict,
    }
)

CONFIG_SCHEMA = vol.Schema(
    {
        vol.Optional(DOMAIN): MATRIX_CHAT_CONFIG_SCHEMA,
    },
    extra=vol.ALLOW_EXTRA,
)

SERVICE_SEND_MESSAGE_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Optional(ATTR_TARGET): cv.string,
        vol.Optional(ATTR_TARGETS): vol.Any(cv.string, [cv.string]),
        vol.Required(ATTR_MESSAGE): cv.string,
        vol.Optional(ATTR_FORMAT, default=FORMAT_TEXT): vol.In([FORMAT_TEXT, FORMAT_HTML]),
        vol.Optional(ATTR_SILENT, default=False): cv.boolean,
        vol.Optional(ATTR_REPLY_TO_EVENT_ID): cv.string,
        vol.Optional(ATTR_EDIT_EVENT_ID): cv.string,
        vol.Optional(ATTR_THREAD_ROOT_EVENT_ID): cv.string,
    }
)

SERVICE_SEND_MEDIA_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Optional(ATTR_TARGET): cv.string,
        vol.Optional(ATTR_TARGETS): vol.Any(cv.string, [cv.string]),
        vol.Required(ATTR_FILE_PATH): cv.string,
        vol.Optional(ATTR_MESSAGE, default=""): cv.string,
        vol.Optional(ATTR_MIME_TYPE, default=""): cv.string,
        vol.Optional(ATTR_AUTO_CONVERT): cv.boolean,
        vol.Optional(ATTR_CONVERT_THRESHOLD_MB): vol.Coerce(float),
        vol.Optional(ATTR_MAX_SIZE_MB): vol.Coerce(float),
        vol.Optional(ATTR_THREAD_ROOT_EVENT_ID): cv.string,
    }
)

SERVICE_SEND_REACTION_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Optional(ATTR_TARGET): cv.string,
        vol.Optional(ATTR_TARGETS): vol.Any(cv.string, [cv.string]),
        vol.Required(ATTR_EVENT_ID): cv.string,
        vol.Required(ATTR_REACTION_KEY): cv.string,
    }
)

SERVICE_REDACT_EVENT_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Optional(ATTR_TARGET): cv.string,
        vol.Optional(ATTR_TARGETS): vol.Any(cv.string, [cv.string]),
        vol.Required(ATTR_EVENT_ID): cv.string,
        vol.Optional(ATTR_REASON, default=""): cv.string,
    }
)

SERVICE_GET_INBOUND_CONFIG_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
    }
)

SERVICE_GET_OUTBOX_STATS_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
    }
)

SERVICE_FLUSH_OUTBOX_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Optional("max_items", default=25): vol.Coerce(int),
    }
)

SERVICE_RESOLVE_TARGET_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Required(ATTR_TARGET): cv.string,
    }
)

SERVICE_LIST_ROOMS_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Optional(ATTR_LIMIT, default=50): vol.Coerce(int),
        vol.Optional(ATTR_INCLUDE_MEMBERS, default=False): cv.boolean,
    }
)

SERVICE_JOIN_ROOM_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Required(ATTR_ROOM_OR_ALIAS): cv.string,
    }
)

SERVICE_INVITE_USER_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Required(ATTR_ROOM_ID): cv.string,
        vol.Required(ATTR_USER_ID): cv.string,
        vol.Optional(ATTR_DRY_RUN, default=False): cv.boolean,
    }
)

SERVICE_ENSURE_DM_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Required(ATTR_USER_ID): cv.string,
    }
)

SERVICE_ENSURE_ROOM_ENCRYPTED_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
        vol.Required(ATTR_ROOM_ID): cv.string,
    }
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Matrix Chat from YAML config (optional import path)."""
    hass.data.setdefault(
        DOMAIN,
        {
            "clients": {},
            "coordinators": {},
            "services_registered": False,
            "notify_registered": False,
            "webhooks": {},
        },
    )

    if DOMAIN in config:
        hass.async_create_task(
            hass.config_entries.flow.async_init(
                DOMAIN,
                context={"source": SOURCE_IMPORT},
                data=config[DOMAIN],
            )
        )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Matrix Chat from config entry."""
    session = aiohttp_client.async_get_clientsession(hass)

    def _get_value(key: str, default: Any = "") -> Any:
        if key in entry.options:
            return entry.options[key]
        return entry.data.get(key, default)

    client = MatrixChatClient(
        hass=hass,
        session=session,
        entry_id=entry.entry_id,
        homeserver=entry.data[CONF_HOMESERVER],
        user_id=entry.data[CONF_USER_ID],
        password=entry.data.get(CONF_PASSWORD, ""),
        access_token=entry.data.get(CONF_ACCESS_TOKEN, ""),
        verify_ssl=entry.data.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL),
        encrypted_webhook_url=_get_value(CONF_ENCRYPTED_WEBHOOK_URL, ""),
        encrypted_webhook_token=_get_value(CONF_ENCRYPTED_WEBHOOK_TOKEN, ""),
        dm_encrypted=_get_value(CONF_DM_ENCRYPTED, DEFAULT_DM_ENCRYPTED),
        auto_convert_video=_get_value(CONF_AUTO_CONVERT_VIDEO, DEFAULT_AUTO_CONVERT_VIDEO),
        video_convert_threshold_mb=_get_value(
            CONF_VIDEO_CONVERT_THRESHOLD_MB, DEFAULT_VIDEO_CONVERT_THRESHOLD_MB
        ),
        max_upload_mb=_get_value(CONF_MAX_UPLOAD_MB, DEFAULT_MAX_UPLOAD_MB),
    )

    try:
        await client.async_initialize()
    except MatrixChatAuthError as err:
        raise ConfigEntryAuthFailed(str(err)) from err
    except MatrixChatConnectionError as err:
        raise ConfigEntryNotReady(str(err)) from err
    except Exception as err:  # noqa: BLE001
        raise ConfigEntryNotReady(str(err)) from err

    current_stored_token = _normalize_token(entry.data.get(CONF_ACCESS_TOKEN, ""))
    if client.access_token and client.access_token != current_stored_token:
        new_data = dict(entry.data)
        new_data[CONF_ACCESS_TOKEN] = client.access_token
        hass.config_entries.async_update_entry(entry, data=new_data)

    hass.data[DOMAIN]["clients"][entry.entry_id] = client
    coordinator = MatrixChatGatewayCoordinator(hass, client)
    hass.data[DOMAIN]["coordinators"][entry.entry_id] = coordinator
    await coordinator.async_config_entry_first_refresh()
    await _async_register_inbound_webhook(hass, entry)
    await _async_register_services(hass)
    await _async_register_notify(hass)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Matrix Chat config entry."""
    clients = hass.data.get(DOMAIN, {}).get("clients", {})
    clients.pop(entry.entry_id, None)
    await _async_unregister_inbound_webhook(hass, entry.entry_id)
    coordinators = hass.data.get(DOMAIN, {}).get("coordinators", {})
    coordinators.pop(entry.entry_id, None)

    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if not clients:
        if hass.services.has_service(DOMAIN, SERVICE_SEND_MESSAGE):
            hass.services.async_remove(DOMAIN, SERVICE_SEND_MESSAGE)
        if hass.services.has_service(DOMAIN, SERVICE_SEND_MEDIA):
            hass.services.async_remove(DOMAIN, SERVICE_SEND_MEDIA)
        if hass.services.has_service(DOMAIN, SERVICE_SEND_REACTION):
            hass.services.async_remove(DOMAIN, SERVICE_SEND_REACTION)
        if hass.services.has_service(DOMAIN, SERVICE_REDACT_EVENT):
            hass.services.async_remove(DOMAIN, SERVICE_REDACT_EVENT)
        if hass.services.has_service(DOMAIN, SERVICE_GET_INBOUND_CONFIG):
            hass.services.async_remove(DOMAIN, SERVICE_GET_INBOUND_CONFIG)
        hass.data[DOMAIN]["services_registered"] = False
        if hass.services.has_service("notify", _NOTIFY_SERVICE_NAME):
            hass.services.async_remove("notify", _NOTIFY_SERVICE_NAME)
        hass.data[DOMAIN]["notify_registered"] = False

    return unload_ok


def _extract_notify_targets(call: ServiceCall) -> list[str]:
    raw = call.data.get(_NOTIFY_TARGET)
    if raw is None:
        return []
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


async def _async_register_notify(hass: HomeAssistant) -> None:
    """Register notify.matrix_chat for Telegram-like notify use."""
    if hass.data[DOMAIN].get("notify_registered"):
        return

    async def _handle_notify(call: ServiceCall) -> None:
        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        targets = _extract_notify_targets(call)
        if not targets:
            raise HomeAssistantError("notify.matrix_chat requires target (list of @user:server, !room:server, #alias:server)")

        message = call.data.get(_NOTIFY_MESSAGE, "")
        data = call.data.get(_NOTIFY_DATA) if isinstance(call.data.get(_NOTIFY_DATA), dict) else {}

        # Attachment support:
        # - data.file_path: single path
        # - data.file_paths: list of paths
        # If present, we send media with 'message' as caption.
        file_paths: list[str] = []
        if isinstance(data.get(_NOTIFY_DATA_FILE_PATHS), list):
            for p in data.get(_NOTIFY_DATA_FILE_PATHS):
                if isinstance(p, str) and p.strip():
                    file_paths.append(p.strip())
        elif isinstance(data.get(_NOTIFY_DATA_FILE_PATH), str) and data.get(_NOTIFY_DATA_FILE_PATH).strip():
            file_paths.append(data.get(_NOTIFY_DATA_FILE_PATH).strip())

        if file_paths:
            mime_type = str(data.get(_NOTIFY_DATA_MIME_TYPE) or "").strip()
            auto_convert = bool(
                data.get(_NOTIFY_DATA_AUTO_CONVERT, client.auto_convert_video)
            )
            convert_threshold_mb = float(
                data.get(_NOTIFY_DATA_CONVERT_THRESHOLD_MB, client.video_convert_threshold_mb)
            )
            max_size_mb = float(data.get(_NOTIFY_DATA_MAX_SIZE_MB, client.max_upload_mb))

            for fp in file_paths:
                await client.async_send_media(
                    targets=targets,
                    file_path=fp,
                    message=message or "",
                    mime_type=mime_type,
                    auto_convert=auto_convert,
                    convert_threshold_mb=convert_threshold_mb,
                    max_size_mb=max_size_mb,
                    thread_root_event_id=str(
                        data.get(_NOTIFY_DATA_THREAD_ROOT_EVENT_ID, "") or ""
                    ).strip(),
                )
            return

        await client.async_send_message(
            targets=targets,
            message=message,
            message_format=FORMAT_TEXT,
            silent=bool(data.get(_NOTIFY_DATA_SILENT, False)),
            thread_root_event_id=str(
                data.get(_NOTIFY_DATA_THREAD_ROOT_EVENT_ID, "") or ""
            ).strip(),
        )

    hass.services.async_register(
        "notify",
        _NOTIFY_SERVICE_NAME,
        _handle_notify,
        schema=NOTIFY_SERVICE_SCHEMA,
    )
    hass.data[DOMAIN]["notify_registered"] = True


async def _async_unregister_inbound_webhook(
    hass: HomeAssistant, entry_id: str
) -> None:
    webhooks: dict[str, dict[str, Any]] = hass.data.get(DOMAIN, {}).get("webhooks", {})
    info = webhooks.pop(entry_id, None)
    if info:
        webhook_id = info.get("webhook_id")
        if webhook_id:
            webhook.async_unregister(hass, webhook_id)


async def _async_register_inbound_webhook(hass: HomeAssistant, entry: ConfigEntry) -> None:
    webhooks: dict[str, dict[str, Any]] = hass.data[DOMAIN]["webhooks"]
    await _async_unregister_inbound_webhook(hass, entry.entry_id)

    inbound_enabled = bool(
        entry.options.get(
            CONF_INBOUND_ENABLED, entry.data.get(CONF_INBOUND_ENABLED, DEFAULT_INBOUND_ENABLED)
        )
    )
    shared_secret = str(
        entry.options.get(
            CONF_INBOUND_SHARED_SECRET, entry.data.get(CONF_INBOUND_SHARED_SECRET, "")
        )
        or ""
    ).strip()

    if not inbound_enabled:
        return

    webhook_id = f"{DOMAIN}_{entry.entry_id.replace('-', '')}_inbound"
    webhook.async_register(
        hass,
        DOMAIN,
        f"Matrix Chat Inbound {entry.title}",
        webhook_id,
        _async_handle_inbound_webhook,
        allowed_methods=("POST",),
    )
    webhooks[entry.entry_id] = {
        "webhook_id": webhook_id,
        "shared_secret": shared_secret,
        "enabled": True,
    }

    _LOGGER.info(
        "Matrix Chat inbound webhook registered for entry %s at path %s",
        entry.entry_id,
        webhook.async_generate_path(webhook_id),
    )


async def _async_handle_inbound_webhook(
    hass: HomeAssistant, webhook_id: str, request: web.Request
) -> web.Response:
    webhooks: dict[str, dict[str, Any]] = hass.data.get(DOMAIN, {}).get("webhooks", {})
    entry_id: str | None = None
    info: dict[str, Any] | None = None
    for candidate_entry_id, candidate_info in webhooks.items():
        if candidate_info.get("webhook_id") == webhook_id:
            entry_id = candidate_entry_id
            info = candidate_info
            break

    if entry_id is None or info is None:
        return web.json_response({"error": "unknown_webhook"}, status=404)

    expected_secret = str(info.get("shared_secret") or "").strip()
    if expected_secret:
        provided_secret = request.headers.get("X-Matrix-Chat-Secret", "").strip()
        if provided_secret != expected_secret:
            return web.json_response({"error": "forbidden"}, status=403)

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response({"error": "invalid_json"}, status=400)

    if not isinstance(payload, dict):
        return web.json_response({"error": "payload_must_be_object"}, status=400)

    event_data = dict(payload)
    event_data["entry_id"] = entry_id
    event_data["received_at"] = dt_util.utcnow().isoformat()
    hass.bus.async_fire(EVENT_INBOUND_MESSAGE, event_data)

    _LOGGER.info(
        "Inbound Matrix event fired (%s): sender=%s room=%s event=%s",
        EVENT_INBOUND_MESSAGE,
        event_data.get("sender"),
        event_data.get("room_id"),
        event_data.get("event_id"),
    )

    # Optional: translate inbound Matrix messages into HA service calls.
    # This is disabled by default and must be explicitly allowlisted.
    try:
        await _async_maybe_dispatch_inbound_command(hass, entry_id, event_data)
    except Exception:  # noqa: BLE001
        _LOGGER.exception("Inbound command dispatch failed")
    return web.json_response({"status": "ok"})


def _parse_allowlist(value: Any) -> list[str]:
    """Parse allowlist value from options/data.

    Accepts:
    - list[str]
    - comma-separated string
    - newline-separated string
    """
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out

    raw = str(value).strip()
    if not raw:
        return []
    parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts.extend([p.strip() for p in line.split(",") if p.strip()])
    deduped: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _service_is_allowed(service: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    for pat in patterns:
        if fnmatch.fnmatchcase(service, pat):
            return True
    return False


async def _async_maybe_dispatch_inbound_command(
    hass: HomeAssistant, entry_id: str, event_data: dict[str, Any]
) -> None:
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None:
        return

    enabled = bool(
        entry.options.get(
            CONF_COMMANDS_ENABLED, entry.data.get(CONF_COMMANDS_ENABLED, DEFAULT_COMMANDS_ENABLED)
        )
    )
    if not enabled:
        return

    sender = str(event_data.get("sender") or "").strip()
    room_id = str(event_data.get("room_id") or "").strip()
    msgtype = str(event_data.get("msgtype") or "").strip()

    content = event_data.get("content") if isinstance(event_data.get("content"), dict) else {}
    body = event_data.get("body")
    if body is None and isinstance(content, dict):
        body = content.get("body")
    body = str(body or "")

    if not sender or not room_id:
        return
    if msgtype != "m.text":
        return

    # Allowlists may be configured via YAML import (entry.data) or via options UI
    # (entry.options). Prefer options when set, but always fall back to data.
    allowed_senders = _parse_allowlist(
        entry.options.get(
            CONF_COMMANDS_ALLOWED_SENDERS, entry.data.get(CONF_COMMANDS_ALLOWED_SENDERS, "")
        )
    )
    allowed_rooms = _parse_allowlist(
        entry.options.get(
            CONF_COMMANDS_ALLOWED_ROOMS, entry.data.get(CONF_COMMANDS_ALLOWED_ROOMS, "")
        )
    )
    allowed_services = _parse_allowlist(
        entry.options.get(
            CONF_COMMANDS_ALLOWED_SERVICES, entry.data.get(CONF_COMMANDS_ALLOWED_SERVICES, "")
        )
    )

    # Safety: require sender + service allowlists to be non-empty.
    if not allowed_senders or not allowed_services:
        return
    if sender not in allowed_senders:
        return
    if allowed_rooms and room_id not in allowed_rooms:
        return

    stripped = body.strip()
    if not stripped.startswith(_COMMAND_PREFIX):
        return

    parts = stripped.split(None, 2)
    if len(parts) < 2:
        return

    service_name = parts[1].strip()
    if not _SERVICE_RE.match(service_name):
        await _select_client(hass, entry_id).async_send_message(
            targets=[room_id],
            message=f"Command rejected: invalid service '{service_name}'",
            message_format=FORMAT_TEXT,
        )
        return

    if not _service_is_allowed(service_name, allowed_services):
        await _select_client(hass, entry_id).async_send_message(
            targets=[room_id],
            message=f"Command rejected: service not allowlisted '{service_name}'",
            message_format=FORMAT_TEXT,
        )
        return

    payload_raw = parts[2].strip() if len(parts) >= 3 else "{}"
    try:
        service_data = json.loads(payload_raw) if payload_raw else {}
    except json.JSONDecodeError:
        await _select_client(hass, entry_id).async_send_message(
            targets=[room_id],
            message="Command rejected: JSON parse error (expected: !ha domain.service {\"key\":\"value\"})",
            message_format=FORMAT_TEXT,
        )
        return

    if service_data is None:
        service_data = {}
    if not isinstance(service_data, dict):
        await _select_client(hass, entry_id).async_send_message(
            targets=[room_id],
            message="Command rejected: JSON must be an object",
            message_format=FORMAT_TEXT,
        )
        return

    domain, service = service_name.split(".", 1)
    _LOGGER.warning(
        "Inbound command accepted: sender=%s room=%s service=%s",
        sender,
        room_id,
        service_name,
    )

    try:
        await hass.services.async_call(domain, service, service_data, blocking=True)
        await _select_client(hass, entry_id).async_send_message(
            targets=[room_id],
            message=f"Command ok: {service_name}",
            message_format=FORMAT_TEXT,
        )
    except Exception as err:  # noqa: BLE001
        await _select_client(hass, entry_id).async_send_message(
            targets=[room_id],
            message=f"Command failed: {service_name} ({type(err).__name__})",
            message_format=FORMAT_TEXT,
        )


def _extract_targets(service_data: dict[str, Any]) -> list[str]:
    targets: list[str] = []

    single = service_data.get(ATTR_TARGET)
    if isinstance(single, str) and single.strip():
        targets.append(single.strip())

    many = service_data.get(ATTR_TARGETS)
    if isinstance(many, str) and many.strip():
        parts = [part.strip() for part in many.split(",") if part.strip()]
        targets.extend(parts)
    elif isinstance(many, list):
        for item in many:
            if isinstance(item, str) and item.strip():
                targets.append(item.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for target in targets:
        if target not in seen:
            seen.add(target)
            deduped.append(target)
    return deduped


def _select_client(hass: HomeAssistant, entry_id: str | None) -> MatrixChatClient:
    clients: dict[str, MatrixChatClient] = hass.data.get(DOMAIN, {}).get("clients", {})
    if not clients:
        raise HomeAssistantError("No Matrix Chat config entries are loaded")

    if entry_id:
        client = clients.get(entry_id)
        if client is None:
            raise HomeAssistantError(f"Unknown matrix_chat entry_id: {entry_id}")
        return client

    if len(clients) == 1:
        return next(iter(clients.values()))

    raise HomeAssistantError(
        "Multiple Matrix Chat entries loaded. Provide entry_id in service data."
    )


async def _async_register_services(hass: HomeAssistant) -> None:
    if hass.data[DOMAIN].get("services_registered"):
        return

    async def _handle_send_message(call: ServiceCall) -> ServiceResponse:
        targets = _extract_targets(call.data)
        if not targets:
            raise HomeAssistantError("Provide target or targets")
        if call.data.get(ATTR_REPLY_TO_EVENT_ID) and call.data.get(ATTR_EDIT_EVENT_ID):
            raise HomeAssistantError(
                "reply_to_event_id and edit_event_id cannot be used together"
            )
        if call.data.get(ATTR_THREAD_ROOT_EVENT_ID) and call.data.get(ATTR_EDIT_EVENT_ID):
            raise HomeAssistantError(
                "thread_root_event_id and edit_event_id cannot be used together"
            )

        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        result = await client.async_send_message(
            targets=targets,
            message=call.data[ATTR_MESSAGE],
            message_format=call.data.get(ATTR_FORMAT, FORMAT_TEXT),
            silent=bool(call.data.get(ATTR_SILENT, False)),
            reply_to_event_id=call.data.get(ATTR_REPLY_TO_EVENT_ID, ""),
            edit_event_id=call.data.get(ATTR_EDIT_EVENT_ID, ""),
            thread_root_event_id=call.data.get(ATTR_THREAD_ROOT_EVENT_ID, ""),
        )
        return result

    async def _handle_send_media(call: ServiceCall) -> ServiceResponse:
        targets = _extract_targets(call.data)
        if not targets:
            raise HomeAssistantError("Provide target or targets")

        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        result = await client.async_send_media(
            targets=targets,
            file_path=call.data[ATTR_FILE_PATH],
            message=call.data.get(ATTR_MESSAGE, ""),
            mime_type=call.data.get(ATTR_MIME_TYPE, ""),
            auto_convert=call.data.get(ATTR_AUTO_CONVERT, client.auto_convert_video),
            convert_threshold_mb=call.data.get(
                ATTR_CONVERT_THRESHOLD_MB, client.video_convert_threshold_mb
            ),
            max_size_mb=call.data.get(ATTR_MAX_SIZE_MB, client.max_upload_mb),
            thread_root_event_id=call.data.get(ATTR_THREAD_ROOT_EVENT_ID, ""),
        )
        return result

    async def _handle_send_reaction(call: ServiceCall) -> ServiceResponse:
        targets = _extract_targets(call.data)
        if not targets:
            raise HomeAssistantError("Provide target or targets")

        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        result = await client.async_send_reaction(
            targets=targets,
            event_id=call.data[ATTR_EVENT_ID],
            reaction_key=call.data[ATTR_REACTION_KEY],
        )
        return result

    async def _handle_redact_event(call: ServiceCall) -> ServiceResponse:
        targets = _extract_targets(call.data)
        if not targets:
            raise HomeAssistantError("Provide target or targets")

        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        result = await client.async_redact_event(
            targets=targets,
            event_id=call.data[ATTR_EVENT_ID],
            reason=call.data.get(ATTR_REASON, ""),
        )
        return result

    async def _handle_get_inbound_config(call: ServiceCall) -> ServiceResponse:
        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        webhooks: dict[str, dict[str, Any]] = hass.data.get(DOMAIN, {}).get("webhooks", {})
        info = webhooks.get(client.entry_id, {})
        webhook_id = info.get("webhook_id", "")
        return {
            ATTR_ENTRY_ID: client.entry_id,
            ATTR_INBOUND_ENABLED: bool(info.get("enabled", False)),
            ATTR_INBOUND_EVENT_TYPE: EVENT_INBOUND_MESSAGE,
            ATTR_INBOUND_WEBHOOK_ID: webhook_id,
            ATTR_INBOUND_WEBHOOK_PATH: webhook.async_generate_path(webhook_id)
            if webhook_id
            else "",
        }

    async def _handle_get_outbox_stats(call: ServiceCall) -> ServiceResponse:
        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        stats = await client.async_get_outbox_stats()
        return {
            ATTR_ENTRY_ID: client.entry_id,
            ATTR_OUTBOX_SIZE: stats.get("outbox_size", 0),
            ATTR_OUTBOX_OLDEST_TS: stats.get("outbox_oldest_ts"),
            ATTR_OUTBOX_LAST_ERROR: stats.get("outbox_last_error", ""),
        }

    async def _handle_flush_outbox(call: ServiceCall) -> ServiceResponse:
        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        max_items = int(call.data.get("max_items") or 25)
        stats = await client.async_flush_outbox(max_items=max_items)
        return {
            ATTR_ENTRY_ID: client.entry_id,
            **stats,
        }

    async def _handle_resolve_target(call: ServiceCall) -> ServiceResponse:
        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        resolved = await client.async_resolve_target(call.data[ATTR_TARGET])
        return {ATTR_ENTRY_ID: client.entry_id, **resolved}

    async def _handle_list_rooms(call: ServiceCall) -> ServiceResponse:
        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        limit = int(call.data.get(ATTR_LIMIT) or 50)
        include_members = bool(call.data.get(ATTR_INCLUDE_MEMBERS) or False)
        rooms = await client.async_list_joined_rooms(limit=limit, include_members=include_members)
        return {ATTR_ENTRY_ID: client.entry_id, "rooms": rooms}

    async def _handle_join_room(call: ServiceCall) -> ServiceResponse:
        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        room_id = await client.async_join_room(call.data[ATTR_ROOM_OR_ALIAS])
        return {ATTR_ENTRY_ID: client.entry_id, ATTR_ROOM_ID: room_id}

    async def _handle_invite_user(call: ServiceCall) -> ServiceResponse:
        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        if bool(call.data.get(ATTR_DRY_RUN) or False):
            return {
                ATTR_ENTRY_ID: client.entry_id,
                "status": "dry_run",
                ATTR_ROOM_ID: call.data[ATTR_ROOM_ID],
                ATTR_USER_ID: call.data[ATTR_USER_ID],
            }
        await client.async_invite_user(call.data[ATTR_ROOM_ID], call.data[ATTR_USER_ID])
        return {ATTR_ENTRY_ID: client.entry_id, "status": "ok", ATTR_ROOM_ID: call.data[ATTR_ROOM_ID]}

    async def _handle_ensure_dm(call: ServiceCall) -> ServiceResponse:
        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        room_id = await client.async_ensure_dm(call.data[ATTR_USER_ID])
        encrypted = await client._is_room_encrypted(room_id)
        return {
            ATTR_ENTRY_ID: client.entry_id,
            ATTR_USER_ID: call.data[ATTR_USER_ID],
            ATTR_ROOM_ID: room_id,
            "target_type": "user_dm",
            "encrypted": encrypted,
        }

    async def _handle_ensure_room_encrypted(call: ServiceCall) -> ServiceResponse:
        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        encrypted = await client.async_ensure_room_encrypted(call.data[ATTR_ROOM_ID])
        return {ATTR_ENTRY_ID: client.entry_id, ATTR_ROOM_ID: call.data[ATTR_ROOM_ID], "encrypted": encrypted}

    hass.services.async_register(
        DOMAIN,
        SERVICE_SEND_MESSAGE,
        _handle_send_message,
        schema=SERVICE_SEND_MESSAGE_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_SEND_MEDIA,
        _handle_send_media,
        schema=SERVICE_SEND_MEDIA_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_SEND_REACTION,
        _handle_send_reaction,
        schema=SERVICE_SEND_REACTION_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_REDACT_EVENT,
        _handle_redact_event,
        schema=SERVICE_REDACT_EVENT_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_INBOUND_CONFIG,
        _handle_get_inbound_config,
        schema=SERVICE_GET_INBOUND_CONFIG_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_OUTBOX_STATS,
        _handle_get_outbox_stats,
        schema=SERVICE_GET_OUTBOX_STATS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_FLUSH_OUTBOX,
        _handle_flush_outbox,
        schema=SERVICE_FLUSH_OUTBOX_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RESOLVE_TARGET,
        _handle_resolve_target,
        schema=SERVICE_RESOLVE_TARGET_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_LIST_ROOMS,
        _handle_list_rooms,
        schema=SERVICE_LIST_ROOMS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_JOIN_ROOM,
        _handle_join_room,
        schema=SERVICE_JOIN_ROOM_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_INVITE_USER,
        _handle_invite_user,
        schema=SERVICE_INVITE_USER_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_ENSURE_DM,
        _handle_ensure_dm,
        schema=SERVICE_ENSURE_DM_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_ENSURE_ROOM_ENCRYPTED,
        _handle_ensure_room_encrypted,
        schema=SERVICE_ENSURE_ROOM_ENCRYPTED_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    hass.data[DOMAIN]["services_registered"] = True
