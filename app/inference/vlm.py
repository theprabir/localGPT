import logging
import time
from typing import Generator

from app.config import get_settings
from app.inference.llama_server import (
    LlamaServerError,
    LlamaServerClient,
    LlamaServerStreamAdapter,
    LlamaServerUnavailable,
    LlamaModelUnavailable,
    translate_llama_messages,
)

logger = logging.getLogger("localgpt.inference.vlm")


class VLMUnavailable(RuntimeError):
    pass


class VLMModelUnavailable(RuntimeError):
    pass


class VLM:
    """
    Multimodal VLM gateway for LocalGPT.

    Supported models:
    - smolvlm2-2.2b (primary)
    - qwen3-vl-2b (alternative)

    Selection is driven by VLM_MODEL configuration. llama-server is expected to
    already be running and configured with the selected GGUF bundle.

    This class does not load the VLM into the FastAPI process. It talks to
    llama-server over HTTP using an OpenAI-compatible chat API.
    """

    def __init__(
        self,
        model: str | None = None,
        client: LlamaServerClient | None = None,
    ):
        settings = get_settings()
        self.model = model or settings.vlm_model
        self.client = client or LlamaServerClient()
        self._stream_adapter = LlamaServerStreamAdapter(self.client, self.model)
        self._ready = False

    def health(self) -> bool:
        if not self.client.health():
            return False
        try:
            _ = self.client.chat_completions(
                self.model,
                [{"role": "user", "content": "ok"}],
                max_tokens=1,
            )
            self._ready = True
            return True
        except LlamaModelUnavailable:
            self._ready = False
            return False
        except Exception:
            self._ready = False
            return False

    def is_ready(self) -> bool:
        return self._ready

    def ensure_ready(self) -> bool:
        """
        Readiness check usable on a freshly constructed instance.

        is_ready() only reports a previously cached probe result; every request
        builds a new VLM, so callers must run this instead — otherwise a live
        llama-server would still be reported as unavailable. Uses a lightweight
        liveness probe (GET /), not a token-generating completion.
        """
        if self._ready:
            return True
        self._ready = self.client.health()
        return self._ready

    def describe_status(self) -> dict:
        settings = get_settings()
        return {
            "model": self.model,
            "llama_server_url": settings.llama_server_url,
            "llama_server_reachable": self.client.health(),
            "ready": self._ready,
            "model_friendly_name": _model_friendly_name(self.model),
            "expected_gguf_type": _model_expected_gguf_type(self.model),
        }

    def chat(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str:
        settings = get_settings()
        translated = translate_llama_messages(messages)
        max_tok = max_tokens
        if max_tok is None:
            max_tok = max(settings.vlm_max_context_tokens // 4, 256)

        start = time.monotonic()
        try:
            logger.info(
                "VLM inference start model=%s max_tokens=%s temperature=%.2f",
                self.model,
                max_tok,
                temperature,
            )
            response = self.client.chat_completions(
                self.model,
                translated,
                max_tokens=max_tok,
                temperature=temperature,
            )
            assistant = _extract_assistant_text(response)
            logger.info(
                "VLM inference complete model=%s latency_seconds=%.3f",
                self.model,
                time.monotonic() - start,
            )
            return assistant
        except LlamaModelUnavailable as exc:
            logger.exception("VLM model unavailable: %s", exc)
            raise VLMModelUnavailable(str(exc)) from exc
        except LlamaServerError as exc:
            logger.exception("VLM server error: %s", exc)
            raise VLMUnavailable(str(exc)) from exc
        except Exception as exc:
            logger.exception("VLM inference error: %s", exc)
            raise VLMUnavailable(str(exc)) from exc

    def chat_stream(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> Generator[dict, None, None]:
        settings = get_settings()
        translated = translate_llama_messages(messages)
        max_tok = max_tokens
        if max_tok is None:
            max_tok = max(settings.vlm_max_context_tokens // 4, 256)

        logger.info(
            "VLM streaming inference start model=%s max_tokens=%s temperature=%.2f",
            self.model,
            max_tok,
            temperature,
        )
        try:
            yield from self._stream_adapter.stream_chat(
                translated,
                max_tokens=max_tok,
                temperature=temperature,
            )
        except LlamaModelUnavailable as exc:
            logger.exception("VLM model unavailable during stream: %s", exc)
            raise VLMModelUnavailable(str(exc)) from exc
        except LlamaServerError as exc:
            logger.exception("VLM server error during stream: %s", exc)
            raise VLMUnavailable(str(exc)) from exc
        except Exception as exc:
            logger.exception("VLM streaming inference error: %s", exc)
            raise VLMUnavailable(str(exc)) from exc
        finally:
            logger.info("VLM streaming inference end model=%s", self.model)


def _extract_assistant_text(response: dict) -> str:
    try:
        choices = response.get("choices", [])
        if not choices:
            return ""
        first = choices[0]
        message = first.get("message", {})
        content = message.get("content")
        if content is None:
            return ""
        return str(content)
    except Exception:
        return ""


def _model_friendly_name(model: str) -> str:
    if model == "smolvlm2-2.2b":
        return "SmolVLM2 2.2B"
    if model == "qwen3-vl-2b":
        return "Qwen3-VL 2B"
    return model


def _model_expected_gguf_type(model: str) -> str:
    if model == "smolvlm2-2.2b":
        return "SmolVLM2 2.2B GGUF (multimodal, recommended quantized variant)"
    if model == "qwen3-vl-2b":
        return "Qwen3-VL 2B GGUF (multimodal, recommended quantized variant)"
    return "GGUF multimodal model"


class VisionInferenceError(RuntimeError):
    pass


def build_multimodal_message(
    text: str,
    image_bytes: bytes,
    *,
    image_media_type: str = "image/png",
) -> dict:
    """
    Build a llama-server multimodal message with an embedded image.

    llama-server accepts content blocks where one block is an image. The exact
    accepted image encodings depend on the llama.cpp version in use; Phase 1
    documents PNG/JPEG as supported starting points.
    """
    settings = get_settings()
    acceptable_types = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/gif"}
    if image_media_type not in acceptable_types:
        raise VisionInferenceError(f"Unsupported image media type for VLM inference: {image_media_type}")

    if not image_bytes:
        raise VisionInferenceError("Image bytes are empty")

    # First ensure the image is decodable; this also gives us dimensions for logging.
    try:
        from PIL import Image
        from io import BytesIO

        pil = Image.open(BytesIO(image_bytes))
        w, h = pil.size
        logger.info(
            " VLM image prepared media_type=%s size_bytes=%s width=%s height=%s",
            image_media_type,
            len(image_bytes),
            w,
            h,
        )
    except Exception as exc:
        raise VisionInferenceError("Uploaded image could not be decoded") from exc

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_media_type};base64,{_base64_image(image_bytes)}",
                },
            },
        ],
    }
    return message


def _base64_image(image_bytes: bytes) -> str:
    import base64

    return base64.b64encode(image_bytes).decode("ascii")


def vlm_stream_chat_response(messages: list[dict[str, object]]) -> Generator[dict, None, None]:
    vlm = VLM()
    yield from vlm.chat_stream(messages)


def vlm_chat_response(messages: list[dict[str, object]]) -> str:
    vlm = VLM()
    return vlm.chat(messages)
