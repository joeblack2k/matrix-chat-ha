# Matrix Chat for Home Assistant

`matrix_chat` is a custom Home Assistant integration for Matrix messaging with a bot account.

## Features
- Send text messages to Matrix rooms and users (DM auto-create for `@user:server` targets)
- Send images, videos and files
- Automatic video conversion (MP4 H.264/AAC) above configurable threshold
- Room alias (`#alias:server`) and room ID (`!room:server`) support
- Optional encrypted-room webhook fallback for text messages
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

## Security notes
- Never commit real passwords or access tokens.
- Prefer Home Assistant secrets for credentials.
- The repository intentionally excludes environment and secret files.

## Tested acceptance flow
Validated live on 2026-02-14:
- text message sent
- jpg image sent
- mp4 video sent

All three were delivered successfully via `matrix_chat` services to a real Matrix user target.
