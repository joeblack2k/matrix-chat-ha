"""Matrix Chat diagnostics sensors."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .coordinator import MatrixChatGatewayCoordinator


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    coordinator: MatrixChatGatewayCoordinator = hass.data[DOMAIN]["coordinators"][entry.entry_id]
    async_add_entities(
        [
            MatrixChatGatewayIntSensor(
                coordinator,
                entry_id=entry.entry_id,
                key="inbound_queue_size",
                name="Matrix Chat Gateway Inbound Queue Size",
                icon="mdi:tray-arrow-down",
            ),
            MatrixChatGatewayIntSensor(
                coordinator,
                entry_id=entry.entry_id,
                key="inbound_delivered_total",
                name="Matrix Chat Gateway Inbound Delivered Total",
                icon="mdi:check-circle-outline",
            ),
            MatrixChatGatewayIntSensor(
                coordinator,
                entry_id=entry.entry_id,
                key="inbound_failed_total",
                name="Matrix Chat Gateway Inbound Failed Total",
                icon="mdi:alert-circle-outline",
            ),
            MatrixChatGatewayTimestampSensor(
                coordinator,
                entry_id=entry.entry_id,
                key="inbound_last_success_ts",
                name="Matrix Chat Gateway Inbound Last Success",
                icon="mdi:clock-check-outline",
            ),
        ]
    )


class _BaseGatewaySensor(CoordinatorEntity[MatrixChatGatewayCoordinator], SensorEntity):
    entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        coordinator: MatrixChatGatewayCoordinator,
        *,
        entry_id: str,
        key: str,
        name: str,
        icon: str,
    ) -> None:
        super().__init__(coordinator)
        self._key = key
        self._attr_name = name
        self._attr_icon = icon
        self._attr_unique_id = f"{DOMAIN}:{entry_id}:gateway:{key}"

    @property
    def _value(self) -> Any:
        data = self.coordinator.data or {}
        return data.get(self._key)


class MatrixChatGatewayIntSensor(_BaseGatewaySensor):
    @property
    def native_value(self) -> int | None:
        value = self._value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return None


class MatrixChatGatewayTimestampSensor(_BaseGatewaySensor):
    _attr_device_class = SensorDeviceClass.TIMESTAMP

    @property
    def native_value(self):
        value = self._value
        if value is None:
            return None
        try:
            return dt_util.utc_from_timestamp(float(value))
        except (TypeError, ValueError):
            return None
