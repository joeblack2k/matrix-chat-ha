# How To Examples

## Send text to multiple targets

```yaml
service: matrix_chat.send_message
data:
  targets:
    - "@jeroen:chat.patatje.net"
    - "!serverlogs:chat.patatje.net"
  message: "Deploy complete"
```

## Send image

```yaml
service: matrix_chat.send_media
data:
  target: "!familyroom:chat.patatje.net"
  file_path: "/config/www/doorbell/test_snapshot.jpg"
  message: "Doorbell snapshot"
```

## Send video with conversion enabled

```yaml
service: matrix_chat.send_media
data:
  target: "!familyroom:chat.patatje.net"
  file_path: "/config/www/doorbell_test.mp4"
  auto_convert: true
  convert_threshold_mb: 20
```

## Reply to event

```yaml
service: matrix_chat.send_message
data:
  target: "!room:server"
  message: "Acknowledged"
  reply_to_event_id: "$abc123:server"
```

## Edit message

```yaml
service: matrix_chat.send_message
data:
  target: "!room:server"
  message: "Updated text"
  edit_event_id: "$abc123:server"
```

## Send reaction

```yaml
service: matrix_chat.send_reaction
data:
  target: "!room:server"
  event_id: "$abc123:server"
  reaction_key: "✅"
```

## Use notify.matrix_chat

```yaml
service: notify.matrix_chat
data:
  target:
    - "!room:server"
  message: "Server backup ready"
```

With attachment:

```yaml
service: notify.matrix_chat
data:
  target:
    - "!room:server"
  message: "Snapshot"
  data:
    file_path: "/config/www/snapshot.jpg"
```

## Room helpers

```yaml
service: matrix_chat.ensure_dm
data:
  user_id: "@jeroen:chat.patatje.net"
```

```yaml
service: matrix_chat.ensure_room_encrypted
data:
  room_id: "!room:server"
```
