"""Matrix Chat custom integration."""

from __future__ import annotations

import logging
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
    ATTR_INBOUND_ENABLED,
    ATTR_INBOUND_EVENT_TYPE,
    ATTR_INBOUND_WEBHOOK_ID,
    ATTR_INBOUND_WEBHOOK_PATH,
    ATTR_MAX_SIZE_MB,
    ATTR_MESSAGE,
    ATTR_MIME_TYPE,
    ATTR_REACTION_KEY,
    ATTR_REPLY_TO_EVENT_ID,
    ATTR_TARGET,
    ATTR_TARGETS,
    CONF_ACCESS_TOKEN,
    CONF_AUTO_CONVERT_VIDEO,
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
    SERVICE_SEND_MEDIA,
    SERVICE_SEND_MESSAGE,
    SERVICE_SEND_REACTION,
)

_LOGGER = logging.getLogger(__name__)
PLATFORMS: list[str] = ["sensor", "binary_sensor"]

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
        vol.Optional(CONF_VIDEO_CONVERT_THRESHOLD_MB, default=DEFAULT_VIDEO_CONVERT_THRESHOLD_MB): vol.Coerce(float),
        vol.Optional(CONF_MAX_UPLOAD_MB, default=DEFAULT_MAX_UPLOAD_MB): vol.Coerce(float),
        vol.Optional(CONF_INBOUND_ENABLED, default=DEFAULT_INBOUND_ENABLED): cv.boolean,
        vol.Optional(CONF_INBOUND_SHARED_SECRET, default=""): cv.string,
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
        vol.Optional(ATTR_REPLY_TO_EVENT_ID): cv.string,
        vol.Optional(ATTR_EDIT_EVENT_ID): cv.string,
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

SERVICE_GET_INBOUND_CONFIG_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
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
        if hass.services.has_service(DOMAIN, SERVICE_GET_INBOUND_CONFIG):
            hass.services.async_remove(DOMAIN, SERVICE_GET_INBOUND_CONFIG)
        hass.data[DOMAIN]["services_registered"] = False

    return unload_ok


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
    return web.json_response({"status": "ok"})


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

        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        result = await client.async_send_message(
            targets=targets,
            message=call.data[ATTR_MESSAGE],
            message_format=call.data.get(ATTR_FORMAT, FORMAT_TEXT),
            reply_to_event_id=call.data.get(ATTR_REPLY_TO_EVENT_ID, ""),
            edit_event_id=call.data.get(ATTR_EDIT_EVENT_ID, ""),
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
        SERVICE_GET_INBOUND_CONFIG,
        _handle_get_inbound_config,
        schema=SERVICE_GET_INBOUND_CONFIG_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    hass.data[DOMAIN]["services_registered"] = True
