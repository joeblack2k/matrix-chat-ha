"""Config flow for Matrix Chat."""

from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_PASSWORD
from homeassistant.core import callback
from homeassistant.helpers import aiohttp_client, config_validation as cv

from .client import MatrixChatAuthError, MatrixChatConnectionError, async_validate_credentials
from .const import (
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
)

_LOGGER = logging.getLogger(__name__)


def _unique_id(homeserver: str, user_id: str) -> str:
    return f"{homeserver.rstrip('/').lower()}|{user_id.lower()}"


class MatrixChatConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle Matrix Chat config flow."""

    VERSION = 1

    async def _validate_input(self, data: dict) -> dict[str, str]:
        session = aiohttp_client.async_get_clientsession(self.hass)
        return await async_validate_credentials(
            session=session,
            homeserver=data[CONF_HOMESERVER],
            user_id=data[CONF_USER_ID],
            password=data.get(CONF_PASSWORD, ""),
            access_token=data.get(CONF_ACCESS_TOKEN, ""),
            verify_ssl=data.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL),
        )

    async def async_step_user(self, user_input=None):
        errors: dict[str, str] = {}

        if user_input is not None:
            if not user_input.get(CONF_PASSWORD) and not user_input.get(CONF_ACCESS_TOKEN):
                errors["base"] = "missing_auth"
            else:
                try:
                    auth = await self._validate_input(user_input)
                    await self.async_set_unique_id(
                        _unique_id(user_input[CONF_HOMESERVER], auth["user_id"])
                    )
                    self._abort_if_unique_id_configured()

                    user_input = dict(user_input)
                    user_input[CONF_ACCESS_TOKEN] = auth["access_token"]
                    user_input[CONF_USER_ID] = auth["user_id"]

                    return self.async_create_entry(
                        title=auth["user_id"],
                        data=user_input,
                    )
                except MatrixChatAuthError:
                    errors["base"] = "invalid_auth"
                except MatrixChatConnectionError:
                    errors["base"] = "cannot_connect"
                except Exception:  # noqa: BLE001
                    _LOGGER.exception("Unexpected error in Matrix Chat config flow")
                    errors["base"] = "cannot_connect"

        schema = vol.Schema(
            {
                vol.Required(CONF_HOMESERVER, default="https://matrix.example.org"): cv.url,
                vol.Required(CONF_USER_ID, default="@mybot:matrix.example.org"): cv.string,
                vol.Optional(CONF_PASSWORD, default=""): cv.string,
                vol.Optional(CONF_ACCESS_TOKEN, default=""): cv.string,
                vol.Optional(CONF_VERIFY_SSL, default=DEFAULT_VERIFY_SSL): cv.boolean,
                vol.Optional(CONF_ENCRYPTED_WEBHOOK_URL, default=""): cv.string,
                vol.Optional(CONF_ENCRYPTED_WEBHOOK_TOKEN, default=""): cv.string,
                vol.Optional(CONF_DM_ENCRYPTED, default=DEFAULT_DM_ENCRYPTED): cv.boolean,
                vol.Optional(CONF_AUTO_CONVERT_VIDEO, default=DEFAULT_AUTO_CONVERT_VIDEO): cv.boolean,
                vol.Optional(
                    CONF_VIDEO_CONVERT_THRESHOLD_MB,
                    default=DEFAULT_VIDEO_CONVERT_THRESHOLD_MB,
                ): vol.Coerce(float),
                vol.Optional(CONF_MAX_UPLOAD_MB, default=DEFAULT_MAX_UPLOAD_MB): vol.Coerce(float),
            }
        )

        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    async def async_step_import(self, import_data: dict):
        """Handle import from YAML config."""
        import_data = dict(import_data)
        import_data.setdefault(CONF_PASSWORD, "")
        import_data.setdefault(CONF_ACCESS_TOKEN, "")
        import_data.setdefault(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL)
        import_data.setdefault(CONF_ENCRYPTED_WEBHOOK_URL, "")
        import_data.setdefault(CONF_ENCRYPTED_WEBHOOK_TOKEN, "")
        import_data.setdefault(CONF_DM_ENCRYPTED, DEFAULT_DM_ENCRYPTED)
        import_data.setdefault(CONF_AUTO_CONVERT_VIDEO, DEFAULT_AUTO_CONVERT_VIDEO)
        import_data.setdefault(CONF_VIDEO_CONVERT_THRESHOLD_MB, DEFAULT_VIDEO_CONVERT_THRESHOLD_MB)
        import_data.setdefault(CONF_MAX_UPLOAD_MB, DEFAULT_MAX_UPLOAD_MB)

        if not import_data.get(CONF_PASSWORD) and not import_data.get(CONF_ACCESS_TOKEN):
            return self.async_abort(reason="missing_auth")

        try:
            auth = await self._validate_input(import_data)
        except MatrixChatAuthError:
            return self.async_abort(reason="invalid_auth")
        except MatrixChatConnectionError:
            return self.async_abort(reason="cannot_connect")

        await self.async_set_unique_id(_unique_id(import_data[CONF_HOMESERVER], auth["user_id"]))
        self._abort_if_unique_id_configured(updates=import_data)

        import_data[CONF_ACCESS_TOKEN] = auth["access_token"]
        import_data[CONF_USER_ID] = auth["user_id"]

        return self.async_create_entry(title=auth["user_id"], data=import_data)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return MatrixChatOptionsFlow(config_entry)


class MatrixChatOptionsFlow(config_entries.OptionsFlow):
    """Handle Matrix Chat options."""

    def __init__(self, config_entry):
        self._config_entry = config_entry

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        options = self._config_entry.options
        data = self._config_entry.data

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_ENCRYPTED_WEBHOOK_URL,
                    default=options.get(CONF_ENCRYPTED_WEBHOOK_URL, data.get(CONF_ENCRYPTED_WEBHOOK_URL, "")),
                ): cv.string,
                vol.Optional(
                    CONF_ENCRYPTED_WEBHOOK_TOKEN,
                    default=options.get(CONF_ENCRYPTED_WEBHOOK_TOKEN, data.get(CONF_ENCRYPTED_WEBHOOK_TOKEN, "")),
                ): cv.string,
                vol.Optional(
                    CONF_DM_ENCRYPTED,
                    default=options.get(CONF_DM_ENCRYPTED, data.get(CONF_DM_ENCRYPTED, DEFAULT_DM_ENCRYPTED)),
                ): cv.boolean,
                vol.Optional(
                    CONF_AUTO_CONVERT_VIDEO,
                    default=options.get(CONF_AUTO_CONVERT_VIDEO, data.get(CONF_AUTO_CONVERT_VIDEO, DEFAULT_AUTO_CONVERT_VIDEO)),
                ): cv.boolean,
                vol.Optional(
                    CONF_VIDEO_CONVERT_THRESHOLD_MB,
                    default=options.get(
                        CONF_VIDEO_CONVERT_THRESHOLD_MB,
                        data.get(CONF_VIDEO_CONVERT_THRESHOLD_MB, DEFAULT_VIDEO_CONVERT_THRESHOLD_MB),
                    ),
                ): vol.Coerce(float),
                vol.Optional(
                    CONF_MAX_UPLOAD_MB,
                    default=options.get(CONF_MAX_UPLOAD_MB, data.get(CONF_MAX_UPLOAD_MB, DEFAULT_MAX_UPLOAD_MB)),
                ): vol.Coerce(float),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)
