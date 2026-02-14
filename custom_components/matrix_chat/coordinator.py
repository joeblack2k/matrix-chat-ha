"""Gateway diagnostics coordinator for Matrix Chat."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .client import MatrixChatClient
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class MatrixChatGatewayCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Poll the encrypted gateway /health endpoint for diagnostics."""

    def __init__(self, hass: HomeAssistant, client: MatrixChatClient) -> None:
        super().__init__(
            hass,
            logger=_LOGGER,
            name=f"{DOMAIN}_gateway_{client.entry_id}",
            update_interval=timedelta(seconds=30),
        )
        self._client = client

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            return await self._client.async_get_gateway_health()
        except Exception as err:  # noqa: BLE001
            raise UpdateFailed(str(err)) from err

