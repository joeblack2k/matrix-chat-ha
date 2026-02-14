"""Matrix Chat custom integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import CONF_PASSWORD
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.exceptions import ConfigEntryAuthFailed, HomeAssistantError
from homeassistant.helpers import aiohttp_client, config_validation as cv
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers.update_coordinator import ConfigEntryNotReady

from .client import (
    MatrixChatAuthError,
    MatrixChatClient,
    MatrixChatConnectionError,
    _normalize_token,
)
from .const import (
    ATTR_AUTO_CONVERT,
    ATTR_CONVERT_THRESHOLD_MB,
    ATTR_ENTRY_ID,
    ATTR_FILE_PATH,
    ATTR_FORMAT,
    ATTR_MAX_SIZE_MB,
    ATTR_MESSAGE,
    ATTR_MIME_TYPE,
    ATTR_TARGET,
    ATTR_TARGETS,
    CONF_ACCESS_TOKEN,
    CONF_AUTO_CONVERT_VIDEO,
    CONF_DM_ENCRYPTED,
    CONF_ENCRYPTED_WEBHOOK_TOKEN,
    CONF_ENCRYPTED_WEBHOOK_URL,
    CONF_HOMESERVER,
    CONF_MAX_UPLOAD_MB,
    CONF_USER_ID,
    CONF_VERIFY_SSL,
    CONF_VIDEO_CONVERT_THRESHOLD_MB,
    DEFAULT_AUTO_CONVERT_VIDEO,
    DEFAULT_DM_ENCRYPTED,
    DEFAULT_MAX_UPLOAD_MB,
    DEFAULT_VERIFY_SSL,
    DEFAULT_VIDEO_CONVERT_THRESHOLD_MB,
    DOMAIN,
    FORMAT_HTML,
    FORMAT_TEXT,
    SERVICE_SEND_MEDIA,
    SERVICE_SEND_MESSAGE,
)

_LOGGER = logging.getLogger(__name__)

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


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Matrix Chat from YAML config (optional import path)."""
    hass.data.setdefault(DOMAIN, {"clients": {}, "services_registered": False})

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
    await _async_register_services(hass)

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Matrix Chat config entry."""
    clients = hass.data.get(DOMAIN, {}).get("clients", {})
    clients.pop(entry.entry_id, None)

    if not clients:
        if hass.services.has_service(DOMAIN, SERVICE_SEND_MESSAGE):
            hass.services.async_remove(DOMAIN, SERVICE_SEND_MESSAGE)
        if hass.services.has_service(DOMAIN, SERVICE_SEND_MEDIA):
            hass.services.async_remove(DOMAIN, SERVICE_SEND_MEDIA)
        hass.data[DOMAIN]["services_registered"] = False

    return True


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

        client = _select_client(hass, call.data.get(ATTR_ENTRY_ID))
        result = await client.async_send_message(
            targets=targets,
            message=call.data[ATTR_MESSAGE],
            message_format=call.data.get(ATTR_FORMAT, FORMAT_TEXT),
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

    hass.data[DOMAIN]["services_registered"] = True
