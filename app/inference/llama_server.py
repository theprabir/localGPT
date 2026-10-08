import logging
import time
from typing import AsyncGenerator, Generator

import httpx

from app.config import get_settings

logger = logging.getLogger("localgpt.inference.llama_server")


class LlamaServerError(RuntimeError):
    pass


class LlamaServerUnavailable(LlamaServerError):
    pass


class LlamaModelUnavailable(LlamaServerError):
    pass


class LlamaServerClient:
    """
    Thin client for llama-server's OpenAI-compatible API.

    llama.cpp server endpoint conventions used by LocalGPT:
    - /v1/chat/completions for text and multimodal chat
    - /health or root for liveness
    - model name is configured via LLAMA_SERVER_URL / VLM_MODEL settings

    This client is intentionally minimal and tuned for CPU-only quantized inference.
    """

    def __init__(self, base_url: str | None = None, timeout_seconds: float = 900.0):
        settings = get_settings()
        self.base_url = (base_url or settings.llama_server_url).rstrip("/")
        self.timeout = timeout_seconds
        self._client: httpx.Client | None = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def health(self) -> bool:
        try:
            response = self._ensure_client().get("/", timeout=5.0)
            return response.status_code < 500
        except Exception as exc:
            logger.debug("llama-server health check failed: %s", exc)
            return False

    def chat_completions(
        self,
        model: str,
        messages: list[dict[str, object]],
        *,
        stream: bool = False,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> dict:
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        response = self._ensure_client().post("/v1/chat/completions", json=payload)
        if response.status_code == 404:
            raise LlamaModelUnavailable(f"Model {model} not found on llama-server")
        if response.status_code >= 500:
            raise LlamaServerError(f"llama-server error: {response.status_code}: {response.text[:200]}")
        if response.status_code != 200:
            raise LlamaServerError(f"llama-server error: {response.status_code}: {response.text[:200]}")

        return response.json()

    def chat_completions_stream(
        self,
        model: str,
        messages: list[dict[str, object]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Generator[dict, None, None]:
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        response = self._ensure_client().post(
            "/v1/chat/completions",
            json=payload,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )
        if response.status_code == 404:
            raise LlamaModelUnavailable(f"Model {model} not found on llama-server")
        if response.status_code >= 500:
            raise LlamaServerError(f"llama-server error: {response.status_code}: {response.text[:200]}")
        if response.status_code != 200:
            raise LlamaServerError(f"llama-server error: {response.status_code}: {response.text[:200]}")

        for line in response.iter_lines():
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            raw = line[len("data: ") :].strip()
            if raw in ("", "[DONE]"):
                continue
            try:
                yield _parse_sse_chunk(raw)
            except Exception as exc:
                logger.warning("Failed to parse llama-server SSE chunk: %s", exc)


def _parse_sse_chunk(raw: str) -> dict:
    """
    Parse one SSE payload.

    Callers may pass either the full line (`data: {...}`) or the payload after
    the prefix was already removed. Strip a leading `data:` marker if present —
    never split on ':', because the JSON payload contains colons of its own.
    """
    text = raw.strip()
    if text.startswith("data:"):
        text = text[len("data:") :].strip()
    if not text or text == "[DONE]":
        raise ValueError("SSE chunk carries no JSON payload")
    import json

    return json.loads(text)


class LlamaServerStreamAdapter:
    """
    Adapts llama-server streaming output into a consistent token-stream interface
    used by the chat service.
    """

    def __init__(self, client: LlamaServerClient, model: str):
        self.client = client
        self.model = model

    def stream_chat(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Generator[dict, None, None]:
        yield from self.client.chat_completions_stream(
            self.model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )


def translate_llama_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """
    Ensure messages are in an OpenAI-compatible shape expected by llama-server.

    llama.cpp generally accepts the OpenAI format, but this helper makes sure
    role values and content fields are sane before they reach the server.
    """
    translated: list[dict[str, object]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if content is None:
            content = ""
        translated.append({"role": role, "content": content})
    return translated
