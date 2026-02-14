# Matrix E2EE Gateway

Small HTTP gateway for encrypted Matrix send (text + image + video) used by Home Assistant `matrix_chat`.

## Why this exists
Home Assistant container often misses `olm`, which blocks native Matrix E2EE sending.
This gateway runs separately with `matrix-nio[e2e]` and handles encrypted room sends.

## API
- `GET /health`
- `POST /send_text` JSON:
  - `room_id`
  - `message`
  - `format` (`text` or `html`)
  - optional `reply_to_event_id`
  - optional `edit_event_id`
- `POST /send_media` multipart:
  - `room_id`, `msgtype`, `body`, `caption`, `mime_type`, `info`
  - `file` (binary)
- `POST /send_reaction` JSON:
  - `room_id`
  - `event_id`
  - `reaction_key`

All `POST` calls require `Authorization: Bearer <MATRIX_GATEWAY_TOKEN>`.

## Run
```bash
cp .env.example .env
# Fill env values
mkdir -p data/store
docker compose up -d --build
curl -fsS http://127.0.0.1:18081/health
```

## Home Assistant settings
Set these in Matrix Chat config/options:
- `encrypted_webhook_url`: `http://<gateway-host>:18081`
- `encrypted_webhook_token`: `<MATRIX_GATEWAY_TOKEN>`
- `dm_encrypted`: `true`

## Notes
- For access-token auth, gateway needs a device id (auto-discovered via `/account/whoami` when possible).
- Keep `data/store` persistent so encryption sessions survive restarts.
- Do not expose this service publicly.
