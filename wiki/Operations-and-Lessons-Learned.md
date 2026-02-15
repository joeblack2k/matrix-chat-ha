# Operations and Lessons Learned

## Key lessons

1. Media compatibility must be tested in Element X, not just backend success logs.
2. Device-key publishing is mandatory for stable encrypted messaging behavior.
3. Cross-signing is required to avoid owner-verification trust shield warnings.
4. Cloudflare can block key-write endpoints (1010), so local Synapse writes are needed for bootstrap flows.
5. Token drift causes `M_UNKNOWN_TOKEN`; keep HA token state aligned with Synapse active tokens.

## Runbook snippets

- Unknown/deleted device warning:
  - restore historical device keys when needed.
- Device not verified by owner:
  - bootstrap cross-signing and sign bot devices.
- Media send errors:
  - verify event schema and MIME handling.

## Operational policy

- Route automations and logs through Home Assistant `matrix_chat` + `homeassistant_bot`.
- Avoid ad-hoc direct Synapse scripts for normal notification paths.
