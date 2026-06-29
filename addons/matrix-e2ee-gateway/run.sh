#!/bin/bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-/data/options.json}"

# Home Assistant add-ons expose options via /data/options.json.
# Some deployments also provide the values through the environment, so we
# support both paths to keep the gateway startup robust.

eval "$(python3 - <<'PY' "$CONFIG_FILE"
import json
import os
import pathlib
import shlex
import sys

config_path = sys.argv[1]
try:
    with open(config_path, encoding="utf-8") as handle:
        data = json.load(handle)
except FileNotFoundError:
    data = {}


def pick(name: str, default: str = "") -> str:
    env_name = name.upper()
    if os.getenv(env_name):
        return os.getenv(env_name)
    for key in (name, env_name):
        if key in data and data[key] not in (None, ""):
            return str(data[key])
        if key.lower() in data and data[key.lower()] not in (None, ""):
            return str(data[key.lower()])
    return default


def bool_flag(name: str, default: bool = False) -> str:
    value = data.get(name, data.get(name.upper(), default))
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return "1" if value.strip().lower() in {"1", "true", "yes", "on"} else "0"
    return "1" if bool(value) else "0"

values = {
    "MATRIX_HOMESERVER": pick("matrix_homeserver"),
    "MATRIX_USER_ID": pick("matrix_user_id"),
    "MATRIX_ACCESS_TOKEN": pick("matrix_access_token"),
    "MATRIX_PASSWORD": pick("matrix_password"),
    "MATRIX_GATEWAY_TOKEN": pick("matrix_gateway_token"),
    "MATRIX_DEVICE_ID": pick("matrix_device_id"),
    "MATRIX_DEVICE_NAME": pick("matrix_device_name", "Home Assistant Matrix E2EE Gateway"),
    "MATRIX_STORE_PATH": pick("matrix_store_path", "/data/store"),
    "MATRIX_GATEWAY_HOST": pick("matrix_gateway_host", "0.0.0.0"),
    "MATRIX_GATEWAY_PORT": str(data.get("matrix_gateway_port", data.get("MATRIX_GATEWAY_PORT", 8080))),
    "MATRIX_VERIFY_SSL": bool_flag("verify_ssl", True),
    "MATRIX_IGNORE_UNVERIFIED_DEVICES": bool_flag("matrix_ignore_unverified_devices", True),
    "MATRIX_INBOUND_WEBHOOK_URL": pick("matrix_inbound_webhook_url"),
    "MATRIX_INBOUND_SHARED_SECRET": pick("matrix_inbound_shared_secret"),
    "MATRIX_DEBUG_ENDPOINTS": bool_flag("matrix_debug_endpoints", False),
    "LOG_LEVEL": pick("log_level", "INFO"),
}

if not values["MATRIX_HOMESERVER"]:
    raise SystemExit("MATRIX_HOMESERVER is required; set the add-on option matrix_homeserver")

store_path = values["MATRIX_STORE_PATH"]
try:
    pathlib.Path(store_path).mkdir(parents=True, exist_ok=True)
except OSError:
    fallback_path = "/tmp/matrix-chat-store"
    pathlib.Path(fallback_path).mkdir(parents=True, exist_ok=True)
    values["MATRIX_STORE_PATH"] = fallback_path

for key, value in values.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"

exec python /app/gateway.py
