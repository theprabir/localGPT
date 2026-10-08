# API

## Endpoints

- `GET /` — web UI
- `GET /health` — health + VLM status
- `GET /settings` — current configuration
- `GET /models` — supported models and status
- `GET /system/status` — runtime status + resource snapshot

### Chat

- `POST /api/chat` — send a message and receive a non-streamed response
- `POST /api/chat/stream` — send a message and stream the response
- `POST /api/chat/intent` — classify a message into an intent

### Conversations

- `GET /api/conversations`
- `POST /api/conversations`
- `GET /api/conversations/{id}`
- `DELETE /api/conversations/{id}`
- `GET /api/conversations/{id}/messages`
- `GET /api/conversations/{id}/attachments`

### Files

- `POST /api/upload`
- `GET /api/uploads/{filename}`
- `GET /api/generated/{filename}`
- `GET /api/generated/{image_id}`

## Notes

- Chat streaming uses server-sent events in a plain-text stream.
- Image generation is not yet functional in Phase 1; the DB schema and service interface are prepared for Phase 2.
