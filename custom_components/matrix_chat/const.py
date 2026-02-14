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

DEFAULT_VERIFY_SSL = True
DEFAULT_DM_ENCRYPTED = True
DEFAULT_AUTO_CONVERT_VIDEO = True
DEFAULT_VIDEO_CONVERT_THRESHOLD_MB = 20.0
DEFAULT_MAX_UPLOAD_MB = 200.0
DEFAULT_INBOUND_ENABLED = True

SERVICE_SEND_MESSAGE = "send_message"
SERVICE_SEND_MEDIA = "send_media"
SERVICE_SEND_REACTION = "send_reaction"
SERVICE_GET_INBOUND_CONFIG = "get_inbound_config"

ATTR_ENTRY_ID = "entry_id"
ATTR_TARGET = "target"
ATTR_TARGETS = "targets"
ATTR_MESSAGE = "message"
ATTR_FORMAT = "format"
ATTR_FILE_PATH = "file_path"
ATTR_MIME_TYPE = "mime_type"
ATTR_AUTO_CONVERT = "auto_convert"
ATTR_CONVERT_THRESHOLD_MB = "convert_threshold_mb"
ATTR_MAX_SIZE_MB = "max_size_mb"
ATTR_REPLY_TO_EVENT_ID = "reply_to_event_id"
ATTR_EDIT_EVENT_ID = "edit_event_id"
ATTR_EVENT_ID = "event_id"
ATTR_REACTION_KEY = "reaction_key"
ATTR_INBOUND_WEBHOOK_ID = "inbound_webhook_id"
ATTR_INBOUND_WEBHOOK_PATH = "inbound_webhook_path"
ATTR_INBOUND_EVENT_TYPE = "inbound_event_type"
ATTR_INBOUND_ENABLED = "inbound_enabled"

FORMAT_TEXT = "text"
FORMAT_HTML = "html"

EVENT_INBOUND_MESSAGE = "matrix_chat_inbound"

STORAGE_VERSION = 1
