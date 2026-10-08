import logging
import re
from typing import Any

from app.models.schemas import RouterIntentResponse, ValidationError

logger = logging.getLogger("localgpt.core.router")

SUPPORTED_INTENTS = ("CHAT", "IMAGE_ANALYSIS", "IMAGE_GENERATION", "IMAGE_EDIT")

# Heuristic patterns tuned for lightweight CPU-first operation.
# These are intentionally conservative so we can start routing without depending
# on a separate router model. Structured model output is preferred when available.
IMAGE_GENERATION_PATTERNS = re.compile(
    r"\b(create|generate|draw|make|create an image|show me|illustrate|picture of|image of|render)\b",
    re.IGNORECASE,
)
IMAGE_EDIT_PATTERNS = re.compile(
    r"\b(change|edit|modify|make.*red?|make.*blue|make.*green|turn.*into|turn this|replace|remove|erase|inpainting|img2img|image-to-image|make the|swap)\b",
    re.IGNORECASE,
)
VISION_QUESTION_INDICATORS = re.compile(
    r"\b(what is this|what|where|who|describe|identify|ocr|read|explain this|name|count|list)\b",
    re.IGNORECASE,
)

GENERATION_KEYWORDS = (
    "image",
    "picture",
    "draw",
    "generate",
    "create",
    "render",
    "illustration",
    "logo",
    "poster",
)


def classify_intent(user_text: str, *, has_attachment: bool = False, image_request: bool = False) -> RouterIntentResponse:
    text = (user_text or "").strip()
    if not text:
        raise ValueError("Cannot classify empty user text")

    # Explicit image requests (image upload present) take priority for vision analysis.
    if has_attachment:
        # If the user asks a question about the attachment, it is IMAGE_ANALYSIS.
        if VISION_QUESTION_INDICATORS.search(text):
            return RouterIntentResponse(
                intent="IMAGE_ANALYSIS",
                prompt=text,
                raw_text=text,
            )

        # If the text includes editing language tied to the attached image, treat as IMAGE_EDIT.
        if IMAGE_EDIT_PATTERNS.search(text):
            return RouterIntentResponse(
                intent="IMAGE_EDIT",
                prompt=text,
                operation="img2img",
                raw_text=text,
            )

        # Otherwise, attached image + plain text usually means IMAGE_ANALYSIS.
        return RouterIntentResponse(
            intent="IMAGE_ANALYSIS",
            prompt=text,
            raw_text=text,
        )

    # Explicit text-to-image requests.
    if IMAGE_GENERATION_PATTERNS.search(text):
        return RouterIntentResponse(
            intent="IMAGE_GENERATION",
            prompt=text,
            raw_text=text,
        )

    # Editing language without an explicit attached image.
    if IMAGE_EDIT_PATTERNS.search(text):
        return RouterIntentResponse(
            intent="IMAGE_EDIT",
            prompt=text,
            operation="img2img",
            raw_text=text,
        )

    # Default to CHAT.
    return RouterIntentResponse(
        intent="CHAT",
        prompt=text,
        raw_text=text,
    )


def classify_intent_structured(raw_json: dict[str, Any]) -> RouterIntentResponse:
    """
    Validate model-generated structured routing output with Pydantic.

    Never trust model-generated tool parameters without validation.
    """
    try:
        intent = RouterIntentResponse(**raw_json)
    except ValidationError as exc:
        logger.warning("Rejected malformed structured intent: %s", exc)
        raise ValueError("Invalid structured intent from model") from exc

    if intent.intent not in SUPPORTED_INTENTS:
        raise ValueError(f"Unsupported intent: {intent.intent}")

    if intent.intent == "IMAGE_GENERATION" and not intent.prompt:
        raise ValueError("IMAGE_GENERATION requires a prompt")

    if intent.intent == "IMAGE_EDIT":
        if not intent.prompt:
            raise ValueError("IMAGE_EDIT requires a prompt")
        if intent.operation not in ("img2img", "inpainting"):
            intent.operation = "img2img"

    return intent


def extract_image_prompt_from_request(user_text: str) -> str:
    text = (user_text or "").strip()
    if not text:
        raise ValueError("Empty image request")
    return text


def extract_image_edit_params(text: str) -> dict[str, Any]:
    """
    Lightweight extraction for a text-based image-edit request.

    This is a simple heuristic used when llama-server returns free text instead of
    structured JSON. Preferred path is structured model output validated by Pydantic.
    """
    prompt = text.strip()
    strength = 0.55
    match = re.search(r"\b(strength\s*[:=]\s*(0\.\d+|1(?:\.0+)?))\b", prompt, re.IGNORECASE)
    if match:
        try:
            strength = float(match.group(2))
        except Exception:
            pass
    strength = max(0.0, min(1.0, strength))
    return {"prompt": prompt, "strength": strength, "operation": "img2img"}


def detect_inline_image_token_provider(text: str) -> bool:
    """
    Rough check for whether the user's text mentions an image action at all.
    Used mainly for logging/telemetry.
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in GENERATION_KEYWORDS)
