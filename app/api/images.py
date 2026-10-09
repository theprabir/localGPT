import logging
import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

from app.config import get_settings
from app.core.logging import get_logger
from app.core.resource_manager import get_resource_manager
from app.core.security import (
    sanitize_filename,
    safe_storage_filename,
    validate_image_extension,
    validate_image_mime,
    validate_max_image_size,
)
from app.models.database import get_database
from app.models.schemas import (
    AttachmentUploadResponse,
    GeneratedImageJobResult,
    GeneratedImageResponse,
    ImageEditRequest,
    ImageGenerateRequest,
    ImageJobCancelResponse,
    ImageJobStatusResponse,
)
from app.services.image_service import get_image_service

logger = get_logger("localgpt.api.images")

router = APIRouter(prefix="/api", tags=["images"])


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    conversation_id: int = Form(...),
    db=Depends(get_database),
    settings=Depends(get_settings),
):
    """
    Upload an image attachment to a conversation.

    The file is validated for MIME type, extension, and size before storage.
    The stored filename is a UUID-based safe name; the original filename is kept
    only for display and is sanitized.
    """
    if conversation_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid conversation id")

    conv = db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    media_type = (file.content_type or "application/octet-stream").strip()
    try:
        validate_image_mime(media_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        contents = await file.read()
    except Exception as exc:
        logger.warning("Upload read failed: %s", exc)
        raise HTTPException(status_code=400, detail="Could not read uploaded file") from exc

    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum size is {settings.max_upload_size_bytes} bytes",
        )

    ext = validate_image_extension(file.filename or "unknown")
    stored = safe_storage_filename(file.filename or "", extension=ext)
    stored_path = os.path.join(settings.uploads_dir_path, stored)

    try:
        os.makedirs(settings.uploads_dir_path, exist_ok=True)
        with open(stored_path, "wb") as f:
            f.write(contents)
    except Exception as exc:
        logger.exception("Failed to write uploaded image: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store uploaded image") from exc

    width = height = None
    try:
        from PIL import Image

        with Image.open(stored_path) as im:
            width, height = im.size
    except Exception:
        pass

    attachment_id = db.add_attachment(
        conversation_id=conversation_id,
        original_filename=sanitize_filename(file.filename or "upload"),
        stored_filename=stored,
        media_type=media_type,
        size_bytes=len(contents),
        width=width,
        height=height,
    )

    return AttachmentUploadResponse(
        id=attachment_id,
        conversation_id=conversation_id,
        original_filename=sanitize_filename(file.filename or "upload"),
        stored_filename=stored,
        media_type=media_type,
        size_bytes=len(contents),
        width=width,
        height=height,
        url=f"/uploads/{stored}",
    )


@router.get("/uploads/{filename}")
async def serve_upload(filename: str, settings=Depends(get_settings)):
    """
    Serve a stored upload from the uploads directory.

    Path traversal is blocked by resolving the path and checking it against the
    approved directory.
    """
    safe_name = os.path.basename(filename)
    candidate = os.path.join(settings.uploads_dir_path, safe_name)
    if not os.path.exists(candidate):
        raise HTTPException(status_code=404, detail="Upload not found")

    resolved = os.path.realpath(candidate)
    allowed_root = os.path.realpath(settings.uploads_dir_path)
    if not resolved.startswith(allowed_root + os.sep) and resolved != allowed_root:
        raise HTTPException(status_code=404, detail="Upload not found")

    return FileResponse(candidate)


@router.get("/generated/{filename}")
async def serve_generated(filename: str, settings=Depends(get_settings)):
    """
    Serve a generated image or thumbnail.
    """
    safe_name = os.path.basename(filename)
    candidate = os.path.join(settings.generated_dir_path, safe_name)
    if not os.path.exists(candidate):
        candidate = os.path.join(settings.thumbnails_dir_path, safe_name)
        if not os.path.exists(candidate):
            raise HTTPException(status_code=404, detail="Generated image not found")

    resolved = os.path.realpath(candidate)
    allowed_root = os.path.realpath(settings.generated_dir_path)
    allowed_root_thumb = os.path.realpath(settings.thumbnails_dir_path)
    if not (
        resolved.startswith(allowed_root + os.sep)
        or resolved.startswith(allowed_root_thumb + os.sep)
        or resolved == allowed_root
        or resolved == allowed_root_thumb
    ):
        raise HTTPException(status_code=404, detail="Generated image not found")

    return FileResponse(candidate)


@router.get("/generated/{image_id:int}", response_model=GeneratedImageResponse)
def get_generated_image(
    image_id: int,
    db=Depends(get_database),
    settings=Depends(get_settings),
):
    row = db.get_generated_image(image_id)
    if not row:
        raise HTTPException(status_code=404, detail="Generated image not found")
    return GeneratedImageResponse(
        id=row["id"],
        conversation_id=row["conversation_id"],
        job_id=row["job_id"],
        prompt=row["prompt"],
        stored_filename=row["stored_filename"],
        thumbnail_filename=row.get("thumbnail_filename"),
        media_type=row["media_type"],
        width=row["width"],
        height=row["height"],
        status=row["status"],
        error_text=row.get("error_text"),
        created_at=row["created_at"],
        completed_at=row.get("completed_at"),
    )


# ---------------------------------------------------------------------------
# Phase 2: image generation / editing jobs
# ---------------------------------------------------------------------------


def _image_service(
    db=Depends(get_database),
    settings=Depends(get_settings),
):
    return get_image_service(db=db, settings=settings)


def _require_conversation(conversation_id: int, db) -> None:
    if conversation_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid conversation id")
    if not db.get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")


def _require_engine(svc, capability: str) -> None:
    ready, reason = svc.engine.available()
    if not ready:
        raise HTTPException(status_code=503, detail=reason or "The image engine is not available")
    if not svc.engine.capabilities().get(capability, False):
        raise HTTPException(
            status_code=501,
            detail=f"{capability} is not supported by the configured image engine",
        )


@router.post("/image/generate", response_model=GeneratedImageJobResult)
def generate_image(
    request: ImageGenerateRequest,
    db=Depends(get_database),
    settings=Depends(get_settings),
    svc=Depends(_image_service),
):
    """
    Queue a text-to-image (or img2img when ``image_id`` is set) job.

    Returns immediately with a job id; the client polls ``/api/image/jobs/{id}``.
    Generation runs in a worker thread against an external engine process, so a
    failure can never take the API down (AGENTS.md §14/§17).
    """
    _require_conversation(request.conversation_id, db)
    if request.image_id is not None:
        if not db.get_generated_image(request.image_id):
            raise HTTPException(status_code=404, detail="Source image not found")
        if db.get_generated_image(request.image_id)["conversation_id"] != request.conversation_id:
            raise HTTPException(status_code=403, detail="Source image belongs to another conversation")
        _require_engine(svc, "img2img")
    else:
        _require_engine(svc, "txt2img")

    job = svc.start_generation(
        conversation_id=request.conversation_id,
        prompt=request.prompt,
        width=request.width,
        height=request.height,
        steps=request.steps,
        source_image_id=request.image_id,
    )
    return GeneratedImageJobResult(**job)


@router.post("/image/edit", response_model=GeneratedImageJobResult)
def edit_image(
    request: ImageEditRequest,
    db=Depends(get_database),
    settings=Depends(get_settings),
    svc=Depends(_image_service),
):
    """Queue an image-to-image edit of an uploaded or generated image."""
    _require_conversation(request.conversation_id, db)
    _require_engine(svc, "img2img")

    if request.image_id is not None:
        row = db.get_generated_image(request.image_id)
        if not row:
            raise HTTPException(status_code=404, detail="Source image not found")
        if row["conversation_id"] != request.conversation_id:
            raise HTTPException(status_code=403, detail="Source image belongs to another conversation")
    elif request.attachment_id is not None:
        row = db.get_attachment(request.attachment_id)
        if not row:
            raise HTTPException(status_code=404, detail="Source image not found")
        if row["conversation_id"] != request.conversation_id:
            raise HTTPException(status_code=403, detail="Source image belongs to another conversation")

    job = svc.start_generation(
        conversation_id=request.conversation_id,
        prompt=request.prompt,
        width=settings.max_image_size,
        height=settings.max_image_size,
        steps=request.steps,
        source_image_id=request.image_id,
        source_attachment_id=request.attachment_id,
        strength=request.strength,
    )
    return GeneratedImageJobResult(**job)


@router.get("/image/jobs/{job_id}", response_model=ImageJobStatusResponse)
def image_job_status(
    job_id: str,
    db=Depends(get_database),
    settings=Depends(get_settings),
    svc=Depends(_image_service),
):
    """Poll a job: status, friendly error text, and step progress."""
    job = svc.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Image job not found")
    return ImageJobStatusResponse(**job)


@router.post("/image/jobs/{job_id}/cancel", response_model=ImageJobCancelResponse)
def cancel_image_job(
    job_id: str,
    db=Depends(get_database),
    settings=Depends(get_settings),
    svc=Depends(_image_service),
):
    """Cancel a queued or running job. Finished jobs report ``cancelled: false``."""
    job = svc.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Image job not found")
    was_cancellable = svc.cancel_job(job_id)
    refreshed = svc.get_job(job_id) or job
    return ImageJobCancelResponse(
        job_id=job_id,
        status=refreshed.get("status", "failed"),
        cancelled=was_cancellable,
    )
