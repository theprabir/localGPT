import logging
import uuid

from app.core.resource_manager import ResourceManager
from app.models.database import Database

logger = logging.getLogger("localgpt.services.image")


class ImageService:
    """
    Phase 2 image service surface.

    Phase 1 only prepares the job/status DB schema and the service interface.
    Real generation/editing is implemented when the image engine is integrated.
    """

    def __init__(self, db: Database, rm: ResourceManager, generated_dir: str, thumbnails_dir: str):
        self.db = db
        self.rm = rm
        self.generated_dir = generated_dir
        self.thumbnails_dir = thumbnails_dir

    def create_image_job(self, conversation_id: int, prompt: str, width: int = 512, height: int = 512, steps: int = 2, image_id: int | None = None) -> dict:
        job_id = f"img_{uuid.uuid4().hex[:12]}"
        stored = f"{uuid.uuid4().hex}.png"
        stored_path = f"{self.generated_dir}/{stored}"
        thumbnail_path = None

        image_id = self.db.add_generated_image(
            conversation_id=conversation_id,
            job_id=job_id,
            prompt=prompt,
            stored_filename=stored,
            thumbnail_filename=thumbnail_path,
            media_type="image/png",
            width=width,
            height=height,
            status="queued",
        )

        logger.info("Image job created job_id=%s conversation_id=%s", job_id, conversation_id)
        return {
            "job_id": job_id,
            "status": "queued",
            "image_id": image_id,
            "url": f"/generated/{stored}",
        }

    def get_image(self, image_id: int) -> dict | None:
        return self.db.get_generated_image(image_id)

    def update_status(self, image_id: int, status: str, error_text: str | None = None, completed_at: str | None = None) -> None:
        self.db.update_generated_image_status(image_id, status=status, error_text=error_text, completed_at=completed_at)

    def is_image_engine_available(self) -> bool:
        return self.rm.ensure_image_engine_available()

    def mark_image_engine_ready(self) -> None:
        self.rm.mark_image_engine_ready()

    def mark_image_engine_unready(self) -> None:
        self.rm.mark_image_engine_unready()
