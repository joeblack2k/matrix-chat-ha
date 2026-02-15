# Developer Guide

This document explains internals of the Matrix Chat integration and encrypted gateway.

## 1. High-level components

## 1.1 Home Assistant integration

Path: `custom_components/matrix_chat/`

Core files:

- `__init__.py`: setup, service registration, notify service, inbound webhook, command dispatch.
- `client.py`: Matrix API client logic, target resolution, media handling, retry queue.
- `config_flow.py`: UI and import flow.
- `services.yaml`: service definitions exposed in HA UI.
- `sensor.py` and `binary_sensor.py`: health and outbox observability.

## 1.2 Encrypted gateway

Path: `encrypted_gateway/`

Core file:

- `gateway.py`: HTTP API for encrypted send paths with matrix-nio E2E support.

The gateway exists because HA runtime often lacks reliable olm/e2e runtime behavior for production encrypted sends.

## 2. Data flow

## 2.1 Outbound text

1. HA service call enters `matrix_chat.send_message`.
2. Integration resolves target(s) to concrete room IDs.
3. For encrypted contexts, path may route through gateway.
4. Event is sent as `m.room.message`.
5. If send fails with transient errors, message can be queued in outbox.

## 2.2 Outbound media

1. `matrix_chat.send_media` validates file path and upload limits.
2. MIME type is detected or overridden.
3. Large video may be converted (optional) to MP4 H.264/AAC.
4. Encrypted path sends media through gateway with proper payload schema.
5. Optional thumbnail generation is supported in gateway for image/video.

## 2.3 Inbound message webhook

1. Gateway or external relay posts inbound event to HA webhook.
2. Integration fires HA bus event `matrix_chat_inbound`.
3. Optional command dispatcher evaluates `!ha` commands.
4. Dispatcher enforces sender, room, and service allowlists before executing HA services.

## 3. Service surface

Primary service constants (see `const.py`):

- `send_message`
- `send_media`
- `send_reaction`
- `get_inbound_config`
- `get_outbox_stats`
- `flush_outbox`
- `resolve_target`
- `list_rooms`
- `join_room`
- `invite_user`
- `ensure_dm`
- `ensure_room_encrypted`

Notify bridge:

- `notify.matrix_chat`

`notify.matrix_chat` accepts attachment metadata through `data.file_path` or `data.file_paths` and internally calls `send_media`.

## 4. Queue and reliability model

Outbox in `client.py`:

- Persistent store-backed queue (HA `Store` helper).
- Failed outbound requests can be enqueued.
- `flush_outbox` allows explicit replay.
- Coordinator periodically attempts small flush batches for healing.

Recommended for automations:

- Use outbox stats as health signal.
- Alert if queue remains non-zero for sustained periods.

## 5. E2EE gateway internals

Gateway API:

- `GET /health`
- `POST /send_text`
- `POST /send_media`

Security:

- Bearer token via `MATRIX_GATEWAY_TOKEN` required for POST endpoints.

Storage:

- `MATRIX_STORE_PATH=/data/store` keeps encryption state persistent.

Key startup behavior:

- Restores session from access token when configured.
- Explicitly executes `keys_upload()` on startup.
- Handles `LocalProtocolError: No key upload needed` as non-fatal.

## 6. Trust and cross-signing

Element X trust shield warnings are not solved by event transport alone.

Bot user must have:

- cross-signing master key
- self-signing key
- bot devices signed by self-signing key

Verification endpoint:

- `/_matrix/client/v3/keys/query`

Expected:

- `master_keys[user]` exists
- `self_signing_keys[user]` exists
- `device_keys[user][device].signatures[user][ed25519:<self_signing_key>]` exists

Operational caveat:

- Cloudflare may block sensitive key-write endpoints with `1010`.
- Use local Synapse endpoint (example: `http://127.0.0.1:18090`) for cross-signing bootstrap/signature writes.

## 7. Config and secrets

Never commit:

- `.env`
- Matrix passwords
- Matrix access tokens
- Home Assistant `secrets.yaml` values

Repo guardrails:

- Keep `.env.example` only.
- Keep defaults safe.
- Add explicit docs for all required env vars.

## 8. Testing strategy

Recommended layered tests:

1. Static checks
- `python3 -m py_compile` for integration and gateway python files.

2. Service-level tests
- Run `send_message`, `send_media`, `send_reaction` to a disposable room.

3. E2EE verification
- Confirm encrypted events are sent from expected device ID.
- Confirm device keys and cross-signing visibility via `keys/query`.

4. Client verification
- Validate behavior in Element and Element X (text, image preview, video playback, trust shield).

## 9. Release checklist

1. Sync live code into repository export.
2. Remove generated artifacts (`__pycache__`, backups).
3. Validate no secrets changed.
4. Update docs and examples.
5. Execute test matrix:
- text
- image
- video
- inbound command
- retry queue flush
6. Tag and push release.

## 10. Future enhancements

Potential next steps:

- Native cross-signing bootstrap utility module in gateway.
- Optional admin dashboard for room discovery and policy validation.
- Structured telemetry export for outbox and inbound command audit.
