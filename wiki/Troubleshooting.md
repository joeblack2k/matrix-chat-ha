# Troubleshooting

## `M_UNKNOWN_TOKEN`

Cause:
- invalid or stale Matrix token in integration config.

Fix:
- refresh token, update HA config/secrets, restart/reload integration.

## `Encrypted by an unknown or deleted device`

Cause:
- old timeline events from removed device IDs.

Fix:
- restore needed legacy device key records.

## `Encrypted by a device not verified by its owner`

Cause:
- missing cross-signing graph or missing self-signatures.

Fix:
- bootstrap cross-signing (`master`, `self_signing`) and sign bot devices.

## Cloudflare `error code: 1010`

Cause:
- WAF blocks sensitive key-write endpoint calls.

Fix:
- execute cross-signing writes through local Synapse endpoint (example `http://127.0.0.1:18090`).

## Media received but not openable

Cause:
- payload schema or metadata incompatibility.

Fix:
- validate encrypted media event composition and MIME type, retest on Element X.

## Rate limits (`M_LIMIT_EXCEEDED`)

Fix:
- use retry/backoff; do not hammer login/send endpoints.
