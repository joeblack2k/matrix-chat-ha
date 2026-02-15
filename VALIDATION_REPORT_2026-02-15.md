# Validation Report - Matrix Chat V3.1 to V3.5

Date: 2026-02-15
Environment: Home Assistant 2026.2.2 + Synapse + Element/Element X
Entry ID: `01KHET6V445JCH6AZF4ETEH75N`

## Scope

Sequential test-driven validation of:
1. V3.1 DM + E2EE + media compatibility
2. V3.2 inbound command dispatch (allowlist)
3. V3.3 notify platform (`notify.matrix_chat`)
4. V3.4 outbound retry queue
5. V3.5 room/user helper services

## V3.1 - DM + E2EE + media

### Test steps
- `matrix_chat.ensure_dm` for `@jeroen:chat.patatje.net`
- `matrix_chat.ensure_room_encrypted` for `!wtDgbGRhpdFZjtKApX:chat.patatje.net`
- `matrix_chat.send_message` (text)
- `matrix_chat.send_media` (jpg + mp4)

### Expected
- DM resolves to same room.
- Room encryption is true.
- Gateway logs encrypted media with thumbnail generation.

### Result
PASS
- Gateway logs contained:
  - `Media sent ... msgtype=m.image encrypted=True thumbnail=True`
  - `Media sent ... msgtype=m.video encrypted=True thumbnail=True`
  - `RoomEncryptedImage` / `RoomEncryptedVideo` events observed.

## V3.2 - Inbound commands allowlist

### Test steps
- Baseline `input_text.matrix_inbound_test_marker` set to `baseline-v32`.
- POST webhook payload as allowed sender `@jeroen:chat.patatje.net` with command:
  - `!ha input_text.set_value {...}`
- POST webhook payload as non-allowed sender `@melanie:chat.patatje.net` with same command.

### Expected
- Allowed sender updates marker.
- Non-allowed sender does not update marker.

### Result
PASS
- Marker changed to `auth-v32-...` for allowed sender.
- Marker remained unchanged for denied sender.
- Gateway log showed: `HA inbound command reply ... Command ok: input_text.set_value`.

## V3.3 - Notify platform

### Test steps
- `notify.matrix_chat` text message to `@jeroen:chat.patatje.net`.
- `notify.matrix_chat` with `data.file_paths` containing jpg + mp4.

### Expected
- Text via `/send_text`.
- Media via `/send_media` in encrypted DM.

### Result
PASS
- Gateway logs contained `/send_text` and `/send_media` calls.
- Encrypted media event types observed.

## V3.4 - Retry queue

### Initial issue found
`async_flush_outbox` treated replay as sent without validating `failure_count`, which could drop failed items.

### Fix applied
File: `custom_components/matrix_chat/client.py`
- In `async_flush_outbox`, validate returned `success_count` and `failure_count`.
- If `success_count <= 0` or `failure_count > 0`, treat as failure and requeue with backoff.

### Regression test steps
1. Stop gateway container.
2. Send message to force outbox enqueue.
3. Run `flush_outbox` while gateway is still down.
4. Start gateway container.
5. Run `flush_outbox` again.

### Expected
- Step 3: item stays queued, `flush_failed > 0`, `flush_sent = 0`.
- Step 5: queue drains, `flush_sent > 0`, `outbox_size = 0`, `outbox_last_error = ""`.

### Result
PASS
- Down-state flush returned:
  - `flush_sent: 0`
  - `flush_failed: 1`
  - queue remained present.
- After recovery flush returned:
  - `flush_sent: 1`
  - `flush_failed: 0`
  - `outbox_size: 0`
  - `outbox_last_error: ""`.

## V3.5 - Room/user helpers

### Test steps and outcomes
- `matrix_chat.resolve_target(@jeroen...)` -> PASS (returns DM room + encrypted=true)
- `matrix_chat.list_rooms(limit=8)` -> PASS (includes DM room)
- `matrix_chat.ensure_dm(@jeroen...)` -> PASS
- `matrix_chat.ensure_room_encrypted(dm_room)` -> PASS
- `matrix_chat.join_room(dm_room)` -> PASS
- `matrix_chat.invite_user(..., dry_run=true)` -> PASS (`status: dry_run`)

## Conclusion

All V3.1-V3.5 goals passed after applying the V3.4 replay validation fix.

Remaining practical validation is client-side UX confirmation in Element X (open/play behavior on iOS device), which requires device-side interaction.
