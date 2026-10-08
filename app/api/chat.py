import base64
import logging
import os
from typing import AsyncGenerator, Generator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.config import get_settings
from app.core.logging import get_logger
from app.core.resource_manager import get_resource_manager
from app.core.router import (
    classify_intent,
    extract_image_prompt_from_request,
)
from app.core.security import (
    render_markdown,
    validate_image_extension,
    validate_image_mime,
)
from app.inference.vlm import (
    VLM,
    VLMModelUnavailable,
    VLMUnavailable,
    build_multimodal_message,
)
from app.models.database import get_database
from app.models.schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    ConversationResponse,
    ErrorDetail,
    MessageListResponse,
    RouterIntentResponse,
)

logger = get_logger("localgpt.api.chat")

router = APIRouter(prefix="/api", tags=["chat"])


def _current_user_agent(request: Request) -> str:
    return (request.headers.get("user-agent") or "unknown")[:256]


def _stream_vlm_tokens(
    vlm: VLM,
    messages: list[dict[str, object]],
    conversation_id: int,
    db,
    assistant_message_id: int,
) -> Generator[str, None, None]:
    """
    Stream assistant tokens from llama-server and acknowledge each with the DB.
    This keeps the DB representation simple while streaming.
    """
    full: list[str] = []
    try:
        for chunk in vlm.chat_stream(messages):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                full.append(content)
            yield _build_server_event("token", content=content or "")
    except VLMModelUnavailable as exc:
        logger.warning("Streaming aborted: model unavailable: %s", exc)
        yield _build_server_event("error", error="model_unavailable", message=str(exc))
        return
    except VLMUnavailable as exc:
        logger.warning("Streaming aborted: server unavailable: %s", exc)
        yield _build_server_event("error", error="server_unavailable", message=str(exc))
        return
    except Exception as exc:
        logger.exception("Streaming aborted: %s", exc)
        yield _build_server_event("error", error="inference_error", message="Streaming failed")
        return
    finally:
        # Update the placeholder row created before the stream (also runs on
        # client abort, so partial answers are kept). Never insert a second row.
        assembled = "".join(full)
        if assembled:
            try:
                db.update_message_content(assistant_message_id, assembled, assembled)
            except Exception:
                logger.exception("Failed to persist assistant message after streaming")
        else:
            # Aborted before a single token arrived (or the server failed
            # immediately): drop the blank placeholder so conversations never
            # store empty assistant rows.
            try:
                db.delete_message(assistant_message_id)
            except Exception:
                logger.exception("Failed to remove empty assistant placeholder")


def _build_server_event(event: str, **kwargs) -> str:
    import json

    payload = {"event": event, **kwargs}
    return f"data: {json.dumps(payload)}\n\n"


def _event_token(token: str | None) -> str:
    import json

    return f"data: {json.dumps({'event': 'token', 'content': token or ''})}\n\n"


def _event_done() -> str:
    import json

    return f"data: {json.dumps({'event': 'done'})}\n\n"


def _event_error(code: str, message: str) -> str:
    import json

    return f"data: {json.dumps({'event': 'error', 'code': code, 'message': message})}\n\n"


@router.post("/chat", response_model=ChatMessageResponse)
def post_chat(
    request: ChatMessageRequest,
    db=Depends(get_database),
    settings=Depends(get_settings),
):
    """
    Send a chat message and receive a non-streamed assistant response.
    This is retained for simple clients; streaming is preferred for UX.
    """
    vlm = _get_vlm()
    if not vlm.ensure_ready():
        raise HTTPException(status_code=503, detail="VLM is not available")

    conversation_id, new_conversation = _resolve_conversation(request, db)

    attachment_context = _attachment_context(request, db)

    # Persist the user's own text; the attachment marker is only for the VLM.
    # On regenerate the user message already exists and the stale answer is
    # replaced instead of duplicating the turn.
    user_content = request.content

    if request.regenerate:
        # Only remove a dangling answer from the previous attempt — never an
        # answer that belongs to an earlier turn.
        previous = db.get_last_message(conversation_id)
        if previous and previous["role"] == "assistant":
            db.delete_message(previous["id"])
    else:
        user_message_id = db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_content,
            attachment_id=request.attachment_id,
        )

    vlm_text = (
        f"{attachment_context}\n\n{request.content}"
        if attachment_context
        else request.content
    )

    try:
        if attachment_context:
            image_bytes = _get_attached_image_bytes(request, db)
            if image_bytes:
                msg = build_multimodal_message(
                    vlm_text,
                    image_bytes,
                    image_media_type=(request.image_media_type or "image/png"),
                )
                messages = [msg]
            else:
                messages = [{"role": "user", "content": vlm_text}]
        else:
            messages = [
                {"role": "user", "content": vlm_text},
            ]

        response_text = vlm.chat(messages)
    except VLMModelUnavailable:
        raise HTTPException(status_code=503, detail="The model is not available right now")
    except VLMUnavailable:
        raise HTTPException(status_code=503, detail="The AI service is unavailable right now")
    except Exception as exc:
        logger.exception("Chat inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="An internal error occurred")

    assistant_message_id = db.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=response_text,
        raw_content=response_text,
    )

    return ChatMessageResponse(
        id=assistant_message_id,
        role="assistant",
        content=response_text,
        raw_content=response_text,
        attachment_id=None,
        generated_image_id=None,
        created_at="",  # populated by db in real impl; kept minimal here
    )


@router.post("/chat/stream")
def chat_stream(
    request: ChatMessageRequest,
    db=Depends(get_database),
    settings=Depends(get_settings),
    request_obj: Request = None,  # injected by FastAPI
):
    """
    Stream an assistant response token-by-token.

    Conversation handling mirrors POST /api/chat, but the assistant message is
    persisted after streaming.
    """
    vlm = _get_vlm()

    # Persist state BEFORE the readiness check so history survives even when the
    # model is down. On regenerate the user message is reused and the previous
    # assistant answer is removed so it can be replaced.
    conversation_id, new_conversation = _resolve_conversation(request, db)

    attachment_context = _attachment_context(request, db)
    user_content = request.content

    if request.regenerate:
        # Only remove a dangling answer from the previous attempt — never an
        # answer that belongs to an earlier turn.
        previous = db.get_last_message(conversation_id)
        if previous and previous["role"] == "assistant":
            db.delete_message(previous["id"])
    else:
        db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_content,
            attachment_id=request.attachment_id,
        )

    if not vlm.ensure_ready():
        return StreamingResponse(
            iter([_event_error("model_unavailable", "The model is not available right now")]),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"},
        )

    vlm_text = (
        f"{attachment_context}\n\n{request.content}"
        if attachment_context
        else request.content
    )

    try:
        if attachment_context:
            image_bytes = _get_attached_image_bytes(request, db)
            if image_bytes:
                msg = build_multimodal_message(
                    vlm_text,
                    image_bytes,
                    image_media_type=(request.image_media_type or "image/png"),
                )
                messages = [msg]
            else:
                messages = [{"role": "user", "content": vlm_text}]
        else:
            messages = [{"role": "user", "content": vlm_text}]

        assistant_message_id = db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content="",
            raw_content="",
        )

        def generate() -> Generator[str, None, None]:
            yield from _stream_vlm_tokens(vlm, messages, conversation_id, db, assistant_message_id)

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"},
        )
    except VLMModelUnavailable:
        return StreamingResponse(
            iter([_event_error("model_unavailable", "The model is not available right now")]),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"},
        )
    except VLMUnavailable:
        return StreamingResponse(
            iter([_event_error("server_unavailable", "The AI service is unavailable right now")]),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"},
        )
    except Exception as exc:
        logger.exception("Streaming chat failed: %s", exc)
        return StreamingResponse(
            iter([_event_error("inference_error", "Streaming failed")]),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"},
        )


@router.post("/chat/intent")
def chat_intent(request: ChatMessageRequest) -> RouterIntentResponse:
    """
    Classify a user message into an intent without executing inference.

    Useful for UI routing and for future structured model-based routing.
    """
    try:
        intent = classify_intent(request.content, has_attachment=bool(request.attachment_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return intent


def _get_vlm() -> VLM:
    return VLM()


def _resolve_conversation(request: ChatMessageRequest, db) -> tuple[int, bool]:
    if request.conversation_id is not None:
        conv = db.get_conversation(request.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return request.conversation_id, False

    title = request.content.strip()[:80] or "New chat"
    conversation_id = db.create_conversation(title=title)
    return conversation_id, True


def _attachment_context(request: ChatMessageRequest, db) -> str | None:
    if not request.attachment_id:
        return None
    attachment = db.get_attachment(request.attachment_id)
    if not attachment:
        return None
    return (
        f"[Attached image: {attachment['original_filename']}]"
        if attachment
        else None
    )


def _get_attached_image_bytes(request: ChatMessageRequest, db) -> bytes | None:
    if request.image_base64:
        try:
            return base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image data")

    if request.attachment_id:
        attachment = db.get_attachment(request.attachment_id)
        if not attachment:
            return None
        stored = attachment.get("stored_filename")
        if not stored:
            return None
        path = _storage_path_for(stored)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None

    return None


def _storage_path_for(stored_filename: str) -> str | None:
    settings = get_settings()
    for directory in (settings.uploads_dir_path, settings.generated_dir_path, settings.thumbnails_dir_path):
        candidate = os.path.join(directory, stored_filename)
        if os.path.exists(candidate):
            return candidate
    return None
