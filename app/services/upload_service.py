import logging
import os

from app.core.security import sanitize_filename, safe_storage_filename, validate_image_extension, validate_image_mime
from app.models.database import Database

logger = logging.getLogger("localgpt.services.upload")


class UploadService:
    def __init__(self, db: Database, uploads_dir: str):
        self.db = db
        self.uploads_dir = uploads_dir

    def store(self, conversation_id: int, file_bytes: bytes, filename: str, media_type: str, max_size_bytes: int) -> dict:
        if len(file_bytes) > max_size_bytes:
            raise ValueError(f"File too large: {len(file_bytes)} bytes")

        ext = validate_image_extension(filename)
        stored = safe_storage_filename(filename, extension=ext)
        stored_path = os.path.join(self.uploads_dir, stored)

        os.makedirs(self.uploads_dir, exist_ok=True)
        with open(stored_path, "wb") as f:
            f.write(file_bytes)

        width = height = None
        try:
            from PIL import Image

            with Image.open(stored_path) as im:
                width, height = im.size
        except Exception:
            pass

        attachment_id = self.db.add_attachment(
            conversation_id=conversation_id,
            original_filename=sanitize_filename(filename),
            stored_filename=stored,
            media_type=media_type,
            size_bytes=len(file_bytes),
            width=width,
            height=height,
        )

        return {
            "id": attachment_id,
            "conversation_id": conversation_id,
            "original_filename": sanitize_filename(filename),
            "stored_filename": stored,
            "media_type": media_type,
            "size_bytes": len(file_bytes),
            "width": width,
            "height": height,
            "url": f"/uploads/{stored}",
        }
