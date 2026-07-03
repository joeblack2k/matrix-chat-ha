# Matrix E2EE Gateway add-on

This add-on packages the encrypted Matrix gateway as a Home Assistant add-on so it can be installed separately from the HACS integration.

## Why this exists

The Matrix gateway is a runtime service that needs a persistent crypto store and a long-running process. That does not fit well inside a HACS custom integration, so the recommended deployment model is:

- install the Matrix Chat integration from HACS
- install this add-on separately in Home Assistant OS / Supervised
- point the integration at the add-on URL and token

## Required options

Set these values in the add-on configuration:

- `matrix_homeserver`
- `matrix_user_id`
- one of `matrix_access_token` or `matrix_password`
- `matrix_gateway_token`

The add-on will start the same gateway implementation used by the standalone Docker deployment.
