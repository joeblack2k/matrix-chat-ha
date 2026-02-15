# Developer Deep Dive

## Component layout

- `__init__.py`: setup, services, notify, webhook, command dispatch.
- `client.py`: API calls, media processing, queueing, DM and room helpers.
- `config_flow.py`: auth and options flow.
- `coordinator.py`: periodic coordination and queue healing hooks.

## Gateway layout

- `gateway.py`: matrix-nio E2E session handling, send endpoints, key handling.

## Reliability mechanisms

- Persistent outbox queue in integration.
- Manual replay via `flush_outbox`.
- Gateway health endpoint and persistent crypto store.

## Security design

- Service-call command dispatch disabled by default.
- Requires allowlists for sender, room, and service patterns.
- Shared secret support for inbound webhook endpoint.

## E2EE trust model

Warnings in Element X are tied to key trust, not just encryption transport.

Required for bot user:

- device keys
- cross-signing master key
- self-signing key
- self-signatures over bot devices

## Production validation model

- implement -> run targeted test -> inspect logs -> inspect Matrix state -> verify client behavior.
