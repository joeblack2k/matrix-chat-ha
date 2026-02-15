# Operations and Lessons Learned

This document captures production learnings, fixes, and runbook guidance.

## 1. Operational baseline

Environment assumptions used during validation:

- Matrix homeserver: Synapse.
- Primary clients: Element Web and Element X (iOS).
- Bot user: `@homeassistant_bot:chat.patatje.net`.
- HA routes notifications through `matrix_chat` integration.
- Optional E2EE gateway used for encrypted sends.

## 2. Incident class: media arrives but cannot be opened

Symptoms:

- Event appears in timeline.
- Client shows unsupported or attachment placeholder.

Root causes seen:

- Invalid encrypted media payload shape.
- Missing/incorrect metadata for client expectations.

Fix direction:

- Enforce Element-compatible encrypted media event composition.
- Keep media schema strict and deterministic.
- Validate against both Element and Element X clients.

## 3. Incident class: unknown/deleted device warning

Symptoms:

- Timeline warning: encrypted by unknown or deleted device.

Root causes seen:

- Historical events were signed by old device IDs removed from user device set.

Fix direction:

- Restore relevant legacy device key material where historical readability matters.
- Keep active sending on stable runtime device ID.

## 4. Incident class: device not verified by owner

Symptoms:

- Element X warning: `Encrypted by a device not verified by its owner`.

Root cause:

- Bot user had device keys but lacked complete cross-signing graph (`master` and `self_signing`) and/or missing self-signatures on device keys.

Fix direction:

1. Bootstrap cross-signing keys for bot user.
2. Sign active bot device keys using self-signing key.
3. Verify with `/_matrix/client/v3/keys/query`.

Expected verification state:

- `master_keys[user]` exists.
- `self_signing_keys[user]` exists.
- Device signatures contain bot self-signing key signature.

## 5. Cloudflare WAF caveat

Observed:

- Sensitive key-write endpoints blocked with `error code: 1010` when routed over public domain.

Operational rule:

- For cross-signing and similar key-write actions, target local Synapse endpoint (example `http://127.0.0.1:18090`).

## 6. Token hygiene and auth failures

Symptoms:

- HA integration throws `M_UNKNOWN_TOKEN`.

Root causes:

- Token in integration/secrets out of sync with valid Matrix session.

Fix direction:

- Re-authenticate bot account.
- Update token in HA config entry/secrets.
- Restart/reload integration and retest text + media.

## 7. Gateway hardening takeaways

Implemented safeguards:

- Explicit device key upload/verification at startup (`keys_upload`).
- Graceful handling of `No key upload needed` state.
- Persistent store requirement documented.
- Health endpoint and container healthcheck integrated.

## 8. Test protocol used in production fixes

For each change:

1. Implement fix.
2. Run targeted service test (text/image/video).
3. Validate gateway logs.
4. Validate Synapse DB event/device mapping.
5. Validate client-side behavior in Element X.

## 9. Safety rules for public repository

- Never commit live `.env` or token material.
- Keep examples in `.env.example` only.
- Do not include private room IDs/users unless necessary; prefer placeholders in docs.

## 10. Recommended ongoing checks

Daily or after deployments:

- `matrix_chat.send_message` to a test room.
- `matrix_chat.send_media` with jpg and mp4.
- `matrix_chat.get_outbox_stats` equals zero steady-state.
- Gateway health endpoint returns 200.
- `keys/query` still returns master/self-signing for bot user.

## 11. Migration guidance (Telegram to Matrix)

- Replace direct Telegram notifications with `notify.matrix_chat` where possible.
- Keep service wrappers (`script.*`) for room abstraction.
- Adopt allowlist policy for any inbound command execution.
- Ensure encrypted rooms have validated trust state before cutover.
