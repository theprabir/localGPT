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
    GeneratedImageResponse,
)

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
