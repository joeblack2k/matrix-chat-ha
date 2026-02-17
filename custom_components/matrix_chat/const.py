"""Constants for Matrix Chat custom integration."""

from __future__ import annotations

DOMAIN = "matrix_chat"

CONF_HOMESERVER = "homeserver"
CONF_USER_ID = "user_id"
CONF_PASSWORD = "password"
CONF_ACCESS_TOKEN = "access_token"
CONF_VERIFY_SSL = "verify_ssl"
CONF_ENCRYPTED_WEBHOOK_URL = "encrypted_webhook_url"
CONF_ENCRYPTED_WEBHOOK_TOKEN = "encrypted_webhook_token"
CONF_DM_ENCRYPTED = "dm_encrypted"
CONF_AUTO_CONVERT_VIDEO = "auto_convert_video"
CONF_VIDEO_CONVERT_THRESHOLD_MB = "video_convert_threshold_mb"
CONF_MAX_UPLOAD_MB = "max_upload_mb"
CONF_INBOUND_ENABLED = "inbound_enabled"
CONF_INBOUND_SHARED_SECRET = "inbound_shared_secret"
CONF_COMMANDS_ENABLED = "commands_enabled"
CONF_COMMANDS_ALLOWED_SENDERS = "commands_allowed_senders"
CONF_COMMANDS_ALLOWED_ROOMS = "commands_allowed_rooms"
CONF_COMMANDS_ALLOWED_SERVICES = "commands_allowed_services"

DEFAULT_VERIFY_SSL = True
DEFAULT_DM_ENCRYPTED = True
DEFAULT_AUTO_CONVERT_VIDEO = True
DEFAULT_VIDEO_CONVERT_THRESHOLD_MB = 20.0
DEFAULT_MAX_UPLOAD_MB = 200.0
DEFAULT_INBOUND_ENABLED = True
DEFAULT_COMMANDS_ENABLED = False

SERVICE_SEND_MESSAGE = "send_message"
SERVICE_SEND_MEDIA = "send_media"
SERVICE_SEND_REACTION = "send_reaction"
SERVICE_REDACT_EVENT = "redact_event"
SERVICE_GET_INBOUND_CONFIG = "get_inbound_config"
SERVICE_GET_OUTBOX_STATS = "get_outbox_stats"
SERVICE_FLUSH_OUTBOX = "flush_outbox"
SERVICE_RESOLVE_TARGET = "resolve_target"
SERVICE_LIST_ROOMS = "list_rooms"
SERVICE_JOIN_ROOM = "join_room"
SERVICE_INVITE_USER = "invite_user"
SERVICE_ENSURE_DM = "ensure_dm"
SERVICE_ENSURE_ROOM_ENCRYPTED = "ensure_room_encrypted"

ATTR_ENTRY_ID = "entry_id"
ATTR_TARGET = "target"
ATTR_TARGETS = "targets"
ATTR_ROOM_ID = "room_id"
ATTR_ROOM_OR_ALIAS = "room_or_alias"
ATTR_USER_ID = "user_id"
ATTR_ENCRYPTED = "encrypted"
ATTR_INCLUDE_MEMBERS = "include_members"
ATTR_LIMIT = "limit"
ATTR_DRY_RUN = "dry_run"
ATTR_MESSAGE = "message"
ATTR_FORMAT = "format"
ATTR_SILENT = "silent"
ATTR_FILE_PATH = "file_path"
ATTR_MIME_TYPE = "mime_type"
ATTR_AUTO_CONVERT = "auto_convert"
ATTR_CONVERT_THRESHOLD_MB = "convert_threshold_mb"
ATTR_MAX_SIZE_MB = "max_size_mb"
ATTR_REPLY_TO_EVENT_ID = "reply_to_event_id"
ATTR_EDIT_EVENT_ID = "edit_event_id"
ATTR_THREAD_ROOT_EVENT_ID = "thread_root_event_id"
ATTR_EVENT_ID = "event_id"
ATTR_REASON = "reason"
ATTR_REACTION_KEY = "reaction_key"
ATTR_INBOUND_WEBHOOK_ID = "inbound_webhook_id"
ATTR_INBOUND_WEBHOOK_PATH = "inbound_webhook_path"
ATTR_INBOUND_EVENT_TYPE = "inbound_event_type"
ATTR_INBOUND_ENABLED = "inbound_enabled"
ATTR_OUTBOX_SIZE = "outbox_size"
ATTR_OUTBOX_OLDEST_TS = "outbox_oldest_ts"
ATTR_OUTBOX_LAST_ERROR = "outbox_last_error"

FORMAT_TEXT = "text"
FORMAT_HTML = "html"

EVENT_INBOUND_MESSAGE = "matrix_chat_inbound"

STORAGE_VERSION = 1
