from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class ChatMessageRequest(BaseModel):
    conversation_id: int | None = None
    content: str = Field(min_length=1, max_length=10000)
    attachment_id: int | None = None
    image_base64: str | None = Field(default=None, max_length=20_000_000)
    image_media_type: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    # Regenerate the last assistant answer instead of starting a new turn:
    # the user message already exists in history and must not be duplicated.
    regenerate: bool = False

    @field_validator("content")
    @classmethod
    def ensure_not_only_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content must not be empty or only whitespace")
        return value


class ChatMessageResponse(BaseModel):
    id: int
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    raw_content: str | None = None
    attachment_id: int | None = None
    generated_image_id: int | None = None
    created_at: str
    # Phase 2: set when this turn was routed to the image engine instead of the
    # VLM; the client polls GET /api/image/jobs/{id} for progress and the URL.
    image_job_id: str | None = None


class ConversationCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    metadata: str | None = None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]


class MessageListResponse(BaseModel):
    messages: list[ChatMessageResponse]


class AttachmentResponse(BaseModel):
    id: int
    conversation_id: int
    original_filename: str
    stored_filename: str
    media_type: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    created_at: str


class AttachmentUploadResponse(BaseModel):
    id: int
    conversation_id: int
    original_filename: str
    stored_filename: str
    media_type: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    url: str


class GeneratedImageResponse(BaseModel):
    id: int
    conversation_id: int
    job_id: str
    prompt: str
    stored_filename: str
    thumbnail_filename: str | None = None
    media_type: str
    width: int
    height: int
    status: Literal["queued", "generating", "completed", "failed", "cancelled"]
    error_text: str | None = None
    created_at: str
    completed_at: str | None = None


class GeneratedImageJobResult(BaseModel):
    job_id: str
    status: Literal["queued", "generating", "completed", "failed", "cancelled"]
    image_id: int | None = None
    url: str | None = None
    error_text: str | None = None


class ImageGenerateRequest(BaseModel):
    conversation_id: int
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)
    steps: int = Field(default=2, ge=1, le=8)
    image_id: int | None = Field(default=None, description="Existing image to modify (img2img)")

    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, value: int, info) -> int:
        settings = _get_settings_silently()
        max_size = getattr(settings, "max_image_size", 512) if settings else 512
        if value > max_size:
            raise ValueError(f"Dimensions must be <= {max_size}")
        return value


class ImageEditRequest(BaseModel):
    conversation_id: int
    image_id: int | None = None
    attachment_id: int | None = None  # edit an uploaded photo instead
    prompt: str = Field(min_length=1, max_length=4000)
    strength: float = Field(default=0.55, ge=0.0, le=1.0)
    steps: int = Field(default=2, ge=1, le=8)

    @model_validator(mode="after")
    def _require_source_image(self) -> "ImageEditRequest":
        if self.image_id is None and self.attachment_id is None:
            raise ValueError("Provide image_id or attachment_id as the edit source")
        return self


class ImageJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "generating", "completed", "failed", "cancelled"]
    image_id: int | None = None
    conversation_id: int | None = None
    url: str | None = None
    error_text: str | None = None
    progress: dict | None = None
    prompt: str | None = None


class ImageJobCancelResponse(BaseModel):
    job_id: str
    status: str
    cancelled: bool


class GeneratedImageListResponse(BaseModel):
    images: list[GeneratedImageResponse]


class ImageAnalysisRequest(BaseModel):
    conversation_id: int
    attachment_id: int
    question: str = Field(min_length=1, max_length=4000)


class SettingsResponse(BaseModel):
    host: str
    port: int
    log_level: str
    llama_server_url: str
    vlm_model: str
    vlm_model_path: str
    vlm_max_context_tokens: int
    image_model: str
    max_image_size: int
    image_steps: int
    max_upload_size_bytes: int
    app_name: str
    app_creator: str


class SystemStatusResponse(BaseModel):
    running: bool
    llama_server_reachable: bool | None
    vlm_model: str
    memory_mb: float | None
    cpu_percent: float | None
    version: str


class RouterIntentResponse(BaseModel):
    intent: Literal["CHAT", "IMAGE_ANALYSIS", "IMAGE_GENERATION", "IMAGE_EDIT"]
    prompt: str | None = None
    question: str | None = None
    operation: Literal["img2img", "inpainting"] | None = None
    strength: float | None = None
    width: int | None = None
    height: int | None = None
    raw_text: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: str | None = None


def _get_settings_silently():
    try:
        from app.config import get_settings

        return get_settings()
    except Exception:
        return None


def validate_router_intent(raw: dict[str, Any]) -> RouterIntentResponse:
    try:
        return RouterIntentResponse(**raw)
    except ValidationError as exc:
        raise ValueError("Invalid router intent payload") from exc


def parse_chat_prompt(raw_content: str) -> str:
    if not raw_content:
        raise ValueError("Empty prompt")
    return raw_content.strip()


def parse_attached_image_request(content: str, attachment_id: int) -> tuple[str, int]:
    if not content:
        raise ValueError("Attached image request must include a question")
    return content.strip(), attachment_id
