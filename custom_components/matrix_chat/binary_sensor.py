"""Matrix Chat diagnostics binary sensors."""

from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import MatrixChatGatewayCoordinator


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    coordinator: MatrixChatGatewayCoordinator = hass.data[DOMAIN]["coordinators"][entry.entry_id]
    async_add_entities(
        [
            MatrixChatGatewayOnlineBinarySensor(
                coordinator,
                entry_id=entry.entry_id,
            )
        ]
    )


class MatrixChatGatewayOnlineBinarySensor(
    CoordinatorEntity[MatrixChatGatewayCoordinator], BinarySensorEntity
):
    entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: MatrixChatGatewayCoordinator, *, entry_id: str) -> None:
        super().__init__(coordinator)
        self._attr_name = "Matrix Chat Gateway Online"
        self._attr_unique_id = f"{DOMAIN}:{entry_id}:gateway:online"
        self._attr_icon = "mdi:server-network"

    @property
    def is_on(self) -> bool | None:
        data = self.coordinator.data or {}
        status = data.get("status")
        if status is None:
            return None
        return status == "ok"

