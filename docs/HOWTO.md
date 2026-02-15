# How To Use Matrix Chat in Home Assistant

This guide is practical and example-driven so new users can start quickly.

## 1. Prerequisites

- Home Assistant with custom component support.
- Matrix homeserver and bot user credentials.
- Optional encrypted gateway if you want robust E2EE delivery.

## 2. Install integration

Copy folder:

- `custom_components/matrix_chat`

to:

- `/config/custom_components/matrix_chat`

Restart Home Assistant.

## 3. Configure integration

Use UI add-integration flow or YAML import.

Example YAML import block:

```yaml
matrix_chat:
  homeserver: "https://matrix.example.org"
  user_id: "@homeassistant_bot:matrix.example.org"
  password: !secret matrix_bot_password
  access_token: !secret matrix_bot_access_token
  verify_ssl: true
  encrypted_webhook_url: "http://matrix-e2ee-gateway:8080"
  encrypted_webhook_token: !secret matrix_gateway_token
  dm_encrypted: true
  auto_convert_video: true
  video_convert_threshold_mb: 20
  max_upload_mb: 200
  inbound_enabled: true
  inbound_shared_secret: !secret matrix_inbound_shared_secret
  commands_enabled: false
```

Notes:

- You need at least one auth method: password or access token.
- If both are present, integration can refresh token flows more safely.

## 4. Gateway setup (optional, recommended for encrypted rooms)

```bash
cd encrypted_gateway
cp .env.example .env
# set MATRIX_HOMESERVER, MATRIX_USER_ID, MATRIX_PASSWORD or MATRIX_ACCESS_TOKEN, MATRIX_GATEWAY_TOKEN
mkdir -p data/store
docker compose up -d --build
curl -fsS http://127.0.0.1:18081/health
```

## 5. Core service examples

## 5.1 Send text to a user

```yaml
service: matrix_chat.send_message
data:
  target: "@jeroen:chat.patatje.net"
  message: "De voordeur is open."
```

## 5.2 Send text to a room

```yaml
service: matrix_chat.send_message
data:
  target: "!roomid:chat.patatje.net"
  message: "Systeemstatus: alles groen."
```

## 5.3 Send to multiple targets

```yaml
service: matrix_chat.send_message
data:
  targets:
    - "@jeroen:chat.patatje.net"
    - "!roomid:chat.patatje.net"
  message: "Broadcast test"
```

## 5.4 HTML message

```yaml
service: matrix_chat.send_message
data:
  target: "!roomid:chat.patatje.net"
  format: html
  message: "<b>Alert</b>: temperatuur hoog"
```

## 5.5 Reply to an event

```yaml
service: matrix_chat.send_message
data:
  target: "!roomid:chat.patatje.net"
  message: "Mee eens"
  reply_to_event_id: "$event123:chat.patatje.net"
```

## 5.6 Edit an event

```yaml
service: matrix_chat.send_message
data:
  target: "!roomid:chat.patatje.net"
  message: "Gecorrigeerde waarde: 21.5C"
  edit_event_id: "$event123:chat.patatje.net"
```

## 5.7 Send image

```yaml
service: matrix_chat.send_media
data:
  target: "!roomid:chat.patatje.net"
  file_path: "/config/www/doorbell/test_snapshot.jpg"
  message: "Snapshot deurbel"
```

## 5.8 Send video

```yaml
service: matrix_chat.send_media
data:
  target: "!roomid:chat.patatje.net"
  file_path: "/config/www/doorbell_test.mp4"
  message: "Clip deurbel"
  auto_convert: true
  convert_threshold_mb: 20
```

## 5.9 Send reaction

```yaml
service: matrix_chat.send_reaction
data:
  target: "!roomid:chat.patatje.net"
  event_id: "$event123:chat.patatje.net"
  reaction_key: "👍"
```

## 6. Notify platform examples

Telegram-style usage:

```yaml
service: notify.matrix_chat
data:
  target:
    - "@jeroen:chat.patatje.net"
  message: "Wasmachine klaar"
```

With attachment:

```yaml
service: notify.matrix_chat
data:
  target:
    - "!roomid:chat.patatje.net"
  message: "Nieuwe snapshot"
  data:
    file_path: "/config/www/doorbell/test_snapshot.jpg"
```

Multiple attachments:

```yaml
service: notify.matrix_chat
data:
  target:
    - "!roomid:chat.patatje.net"
  message: "Batch upload"
  data:
    file_paths:
      - "/config/www/doorbell/test_snapshot.jpg"
      - "/config/www/doorbell_test.mp4"
```

## 7. Room and user helper examples

Resolve target:

```yaml
service: matrix_chat.resolve_target
data:
  target: "@jeroen:chat.patatje.net"
```

Ensure DM exists:

```yaml
service: matrix_chat.ensure_dm
data:
  user_id: "@jeroen:chat.patatje.net"
```

Join room by alias:

```yaml
service: matrix_chat.join_room
data:
  room_or_alias: "#server-logs:chat.patatje.net"
```

Invite user:

```yaml
service: matrix_chat.invite_user
data:
  room_id: "!roomid:chat.patatje.net"
  user_id: "@newuser:chat.patatje.net"
  dry_run: false
```

Enable encryption in room:

```yaml
service: matrix_chat.ensure_room_encrypted
data:
  room_id: "!roomid:chat.patatje.net"
```

## 8. Inbound webhook and commands

## 8.1 Get inbound webhook settings

```yaml
service: matrix_chat.get_inbound_config
data: {}
```

## 8.2 Listen for inbound events

Automation trigger example:

```yaml
trigger:
  - platform: event
    event_type: matrix_chat_inbound
```

## 8.3 Enable commands safely

Set in options/config:

- `commands_enabled: true`
- `commands_allowed_senders`: explicit sender IDs
- `commands_allowed_rooms`: explicit room IDs
- `commands_allowed_services`: explicit service allowlist patterns

Use smallest possible allowlists.

## 9. Retry queue operations

Get queue stats:

```yaml
service: matrix_chat.get_outbox_stats
data: {}
```

Flush queue:

```yaml
service: matrix_chat.flush_outbox
data:
  max_items: 25
```

## 10. Full automation examples

## 10.1 Doorbell snapshot to family room

```yaml
alias: Doorbell Snapshot Matrix
trigger:
  - platform: state
    entity_id: binary_sensor.doorbell_motion
    to: "on"
action:
  - service: matrix_chat.send_media
    data:
      target: "!familyroom:chat.patatje.net"
      file_path: "/config/www/doorbell/test_snapshot.jpg"
      message: "Deurbel beweging"
mode: restart
```

## 10.2 Server health alert with notify platform

```yaml
alias: Server Health Matrix
trigger:
  - platform: state
    entity_id: binary_sensor.server_unhealthy
    to: "on"
action:
  - service: notify.matrix_chat
    data:
      target:
        - "!serverlogs:chat.patatje.net"
      message: "Server health warning"
mode: single
```

## 11. Troubleshooting quick map

- `M_UNKNOWN_TOKEN`: refresh/update token in integration config/secrets.
- `unknown or deleted device`: restore legacy device keys for historical events.
- `device not verified by owner`: bootstrap cross-signing and sign bot devices.
- Media arrives but cannot open: validate encrypted media schema and mime handling.
- 429 rate limit: retry with backoff, avoid brute retries.

Deep troubleshooting and lessons learned: `docs/OPERATIONS_AND_LESSONS.md`.
