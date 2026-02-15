# Matrix Chat Custom Component for Home Assistant

This repository contains the `matrix_chat` Home Assistant custom component and the optional Matrix E2EE gateway used for encrypted rooms (Element/Element X compatible).

## Status (2026-02-15)

V3.1 through V3.5 are implemented and validated in a live setup.

Implemented optional features:
- Media thumbnails for image/video, including encrypted thumbnail payloads.
- Inbound Matrix `!ha` commands with strict sender/room/service allowlists.
- `notify.matrix_chat` platform with text + attachment support.
- Persistent outbound retry queue with manual flush support.
- Room/user helper services:
  - `matrix_chat.resolve_target`
  - `matrix_chat.list_rooms`
  - `matrix_chat.join_room`
  - `matrix_chat.invite_user` (`dry_run` supported)
  - `matrix_chat.ensure_dm`
  - `matrix_chat.ensure_room_encrypted`

## Core capabilities

- Send text to users (`@user:server`) and rooms (`!room:server`, `#alias:server`)
- Auto-create/reuse DM rooms
- Send image/video/file media
- Reply/edit/reaction support
- Optional video conversion for compatibility (MP4 H.264/AAC)
- Encrypted-room transport via gateway for E2EE delivery

## Repository layout

- `custom_components/matrix_chat/` -> Home Assistant custom integration
- `encrypted_gateway/` -> Matrix E2EE gateway service
- `VALIDATION_REPORT_2026-02-15.md` -> sequential V3.1-V3.5 test report

## Installation

1. Copy `custom_components/matrix_chat` into your HA config directory.
2. Configure integration via UI (recommended) or YAML import.
3. For encrypted rooms, run the gateway from `encrypted_gateway/` and set:
   - `encrypted_webhook_url`
   - `encrypted_webhook_token`

## Security

- Do not commit `.env`, passwords, access tokens, or secrets.
- Use Home Assistant secrets (`!secret`) for credentials.
- Keep gateway token and Matrix credentials private.

## Notes about DM and E2EE

- Matrix DMs are still rooms; there is no protocol-level direct socket chat.
- When E2EE is active, media events must use encrypted `file` objects (no top-level `url`).
- This project enforces that for encrypted media, which improves Element X compatibility.
