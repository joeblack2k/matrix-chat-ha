# Quick Start

## 1. Install component

Copy:

- `custom_components/matrix_chat`

to:

- `/config/custom_components/matrix_chat`

Restart Home Assistant.

## 2. Add integration

Use Home Assistant UI:

- Settings -> Devices & Services -> Add Integration -> Matrix Chat

Provide:

- Homeserver URL
- Bot user ID
- Password or Access Token

## 3. Optional encrypted gateway

```bash
cd encrypted_gateway
cp .env.example .env
mkdir -p data/store
docker compose up -d --build
```

Set in Matrix Chat options:

- `encrypted_webhook_url`
- `encrypted_webhook_token`
- `dm_encrypted=true`

## 4. First send test

```yaml
service: matrix_chat.send_message
data:
  target: "@user:server"
  message: "Matrix Chat is live"
```

## 5. Media test

```yaml
service: matrix_chat.send_media
data:
  target: "!room:server"
  file_path: "/config/www/test.jpg"
  message: "Media test"
```

## 6. Health check

- `matrix_chat.get_outbox_stats`
- gateway `GET /health`
