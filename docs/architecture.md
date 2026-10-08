# LocalGPT Architecture

LocalGPT is a lightweight, local-first multimodal AI assistant. It combines a multimodal VLM with an optional image engine and a Python orchestrator (FastAPI). The frontend is a server-rendered, ChatGPT-like UI built with HTML, CSS, htmx, and Alpine.js.

## Components

- **Frontend**: server-rendered HTML + Alpine.js for small interactive state (theme, sidebar, composer, streaming UI).
- **Backend**: FastAPI + Uvicorn.
- **Persistence**: SQLite for conversations, messages, attachments, generated images, and settings.
- **VLM runtime**: llama-server exposing an OpenAI-compatible API. LocalGPT talks to llama-server over HTTP.
- **Image engine**: planned for Phase 2; the Phase 1 foundation includes job/status schema and service interface.

## Request flow

1. User sends a chat message or uploads an image via the UI.
2. FastAPI receives the request and persists user messages/attachments.
3. For chat/voice, the backend forwards the request to llama-server and streams the response.
4. For image tasks (Phase 2), the router classifies intent and hands off to the image engine.

## Key design decisions

- CPU-only orientation.
- No large ML framework resident in the FastAPI process for VLM inference.
- Configurable model selection (SmolVLM2 2.2B primary, Qwen3-VL 2B alternative).
- Local-only operation with no cloud APIs by default.
