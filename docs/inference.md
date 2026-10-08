# Inference

LocalGPT talks to llama-server over HTTP using an OpenAI-compatible chat endpoint. This keeps the FastAPI process lightweight and avoids loading the VLM into the server process.

## Streaming

Chat responses are streamed token-by-token from llama-server. The API uses a plain-text SSE-style stream.

## Multimodal inputs

Images are embedded in chat messages as base64 data URIs when supported by the llama.cpp version in use.

## LLM runtime

llama-server is the recommended runtime. Start it with a supported SmolVLM2 or Qwen3-VL GGUF and point `LLAMA_SERVER_URL` at it.
