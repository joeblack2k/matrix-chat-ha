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
CONF_AUTO_CONVERT_VIDEO = "auto_convert_video"
CONF_VIDEO_CONVERT_THRESHOLD_MB = "video_convert_threshold_mb"
CONF_MAX_UPLOAD_MB = "max_upload_mb"

DEFAULT_VERIFY_SSL = True
DEFAULT_AUTO_CONVERT_VIDEO = True
DEFAULT_VIDEO_CONVERT_THRESHOLD_MB = 20.0
DEFAULT_MAX_UPLOAD_MB = 200.0

SERVICE_SEND_MESSAGE = "send_message"
SERVICE_SEND_MEDIA = "send_media"

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

FORMAT_TEXT = "text"
FORMAT_HTML = "html"

STORAGE_VERSION = 1
