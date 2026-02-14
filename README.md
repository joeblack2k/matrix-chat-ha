# Matrix Chat for Home Assistant

`matrix_chat` is a custom Home Assistant integration for Matrix messaging with a bot account.

## Features
- Send text messages to Matrix rooms and users (DM auto-create for `@user:server` targets)
- Send images, videos and files
- Automatic video conversion (MP4 H.264/AAC) above configurable threshold
- Room alias (`#alias:server`) and room ID (`!room:server`) support
- Optional encrypted gateway fallback for encrypted rooms (text and media)
- DM room selection via `m.direct` account data (reuses existing DM before creating one)
- Service responses include per-target delivery status and event IDs

## Included visual assets
- `custom_components/matrix_chat/icon.svg`
- `custom_components/matrix_chat/logo.svg`

## Install
Copy `custom_components/matrix_chat` into your Home Assistant config directory.

Example path:
- `/config/custom_components/matrix_chat/`

## Configure (UI)
Add integration **Matrix Chat** and provide:
- Homeserver URL (e.g. `https://chat.example.org`)
- Bot MXID (e.g. `@mybot:example.org`)
- Password (recommended) or access token

## Configure (YAML import)
Optional YAML import path:

```yaml
matrix_chat:
  homeserver: "https://chat.example.org"
  user_id: "@mybot:example.org"
  password: ""
  access_token: "YOUR_ACCESS_TOKEN"
  verify_ssl: true
  auto_convert_video: true
  video_convert_threshold_mb: 20
  max_upload_mb: 200
  encrypted_webhook_url: ""
  encrypted_webhook_token: ""
  dm_encrypted: true
```

## Services
### `matrix_chat.send_message`
- `target` or `targets`
- `message`
- `format` (`text` or `html`)

### `matrix_chat.send_media`
- `target` or `targets`
- `file_path`
- optional `message`, `mime_type`
- optional conversion/upload controls

## Encrypted Rooms (Element X / lock icon)
Home Assistant runtime usually lacks `olm`, so encrypted Matrix send in-process is not reliable.
Use the included encrypted gateway service:

- Source: `encrypted_gateway/`
- Endpoints used by integration:
  - `POST /send_text`
  - `POST /send_media`

Gateway quick start:

```bash
cd encrypted_gateway
cp .env.example .env
# Fill MATRIX_* values (do not commit .env)
docker compose up -d --build
curl -fsS http://127.0.0.1:18081/health
```

Then set in Matrix Chat config/options:
- `encrypted_webhook_url`: `http://<gateway-host>:18081`
- `encrypted_webhook_token`: same as `MATRIX_GATEWAY_TOKEN`

## DM behavior
Matrix does not send user-to-user outside rooms; DMs are still private rooms.
This integration now:
- Reuses existing DM room from `m.direct` when available.
- Creates a new DM only if needed.
- Creates encrypted DM by default (`dm_encrypted: true`).

## Security notes
- Never commit real passwords or access tokens.
- Prefer Home Assistant secrets for credentials.
- The repository intentionally excludes environment and secret files.
- For encrypted-room delivery, configure an encrypted gateway endpoint and token.

## Tested acceptance flow
Validated live on 2026-02-14:
- text message sent
- jpg image sent
- mp4 video sent

All three were delivered successfully via `matrix_chat` services to a real Matrix user target.
