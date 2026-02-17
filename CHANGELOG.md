# Changelog

All notable changes to this project are documented in this file.

## 2026-02-17 - v4.3

### Added

- `matrix_chat.redact_event` service to redact/delete existing Matrix events by `event_id`.
- Thread support for outbound text and media using `thread_root_event_id`.
- `send_message` support for `silent` (`m.notice`) in service schema and `notify.matrix_chat`.

### Changed

- Encrypted gateway now accepts and applies:
  - `silent` on `/send_text`
  - `thread_root_event_id` on `/send_text` and `/send_media`
- Service docs (`services.yaml`) updated with:
  - `redact_event`
  - `thread_root_event_id` fields
  - `silent` field documentation
- Validation guard added: `thread_root_event_id` and `edit_event_id` are mutually exclusive.

### Validated

- End-to-end tests passed in encrypted room flow:
  - text send
  - image send
  - silent message send
  - threaded text + threaded media
  - event redaction

### Notes

- Poll events were intentionally postponed for a dedicated client-rendering validation pass (Element Web + Element X) before release.
