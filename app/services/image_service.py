"""Phase 2 image service: job orchestration around an ImageEngine.

Design constraints from AGENTS.md:
  * §14 — generation must not block the HTTP request; return a job id at once.
  * §15 — only one heavy inference process at a time on the 10 GB target.
  * §17 — the engine runs as a separate process; a failed job must never crash
    FastAPI, and errors shown to the user are friendly (tracebacks stay in logs).
  * §18 — operations go through the ImageEngine interface; unsupported work
    raises ``UnsupportedOperation`` instead of being faked.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.resource_manager import ResourceManager
from app.inference.image_engine import (
    GenerationCancelled,
    ImageEngineError,
    UnsupportedOperation,
)
from app.models.database import Database

logger = logging.getLogger("localgpt.services.image")

FRIENDLY_BUSY = (
    "Image generation could not start: not enough free memory. "
    "Close some programs and try again."
)
FRIENDLY_FAILURE = (
    "Image generation failed. The image engine could not start. "
    "Check the model configuration and system resources."
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ImageService:
    """Creates, runs, monitors and cancels image jobs."""

    def __init__(
        self,
        db: Database,
        rm: ResourceManager,
        generated_dir: str,
        thumbnails_dir: str,
        engine=None,
        settings=None,
    ):
        self.db = db
        self.rm = rm
        self.generated_dir = generated_dir
        self.thumbnails_dir = thumbnails_dir
        self.settings = settings
        self._engine = engine
        self._jobs: dict[str, dict] = {}
        self._jobs_lock = threading.Lock()
        # One heavy inference process at a time (AGENTS.md §14/§15).
        self._run_lock = threading.Lock()

    # -- engine ------------------------------------------------------------

    @property
    def engine(self):
        if self._engine is None:
            from app.inference.image_engine import build_image_engine

            self._engine = build_image_engine(self.settings)
            ready, _ = self._engine.available()
            if ready:
                self.rm.mark_image_engine_ready()
            else:
                self.rm.mark_image_engine_unready()
        return self._engine

    def is_image_engine_available(self) -> bool:
        ready, _ = self.engine.available()
        return ready

    def engine_status(self) -> dict:
        ready, reason = self.engine.available()
        return {
            "ready": ready,
            "engine": getattr(self.engine, "name", "unknown"),
            "reason": reason or None,
            "capabilities": self.engine.capabilities(),
        }

    def mark_image_engine_ready(self) -> None:
        self.rm.mark_image_engine_ready()

    def mark_image_engine_unready(self) -> None:
        self.rm.mark_image_engine_unready()

    # -- job records -------------------------------------------------------

    def create_image_job(
        self,
        conversation_id: int,
        prompt: str,
        width: int = 512,
        height: int = 512,
        steps: int = 2,
        image_id: int | None = None,
    ) -> dict:
        job_id = f"img_{uuid.uuid4().hex[:12]}"
        stored = f"{uuid.uuid4().hex}.png"
        os.makedirs(self.generated_dir, exist_ok=True)

        new_id = self.db.add_generated_image(
            conversation_id=conversation_id,
            job_id=job_id,
            prompt=prompt,
            stored_filename=stored,
            thumbnail_filename=None,
            media_type="image/png",
            width=width,
            height=height,
            status="queued",
        )

        logger.info(
            "Image job created job_id=%s image_id=%s conversation_id=%s source=%s",
            job_id,
            new_id,
            conversation_id,
            image_id,
        )
        return {
            "job_id": job_id,
            "status": "queued",
            "image_id": new_id,
            "url": f"/api/generated/{stored}",
        }

    # -- lifecycle ---------------------------------------------------------

    def start_generation(
        self,
        conversation_id: int,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        steps: int = 2,
        negative_prompt: str = "",
        seed: int = -1,
        source_image_id: int | None = None,
        source_attachment_id: int | None = None,
        strength: float | None = None,
        mask_path: str | None = None,
    ) -> dict:
        """Queue a txt2img / img2img / inpaint job; returns immediately."""
        job = self.create_image_job(
            conversation_id=conversation_id,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            image_id=source_image_id,
        )
        cancel = threading.Event()
        with self._jobs_lock:
            self._jobs[job["job_id"]] = {
                "cancel": cancel,
                "progress": None,
                "running": False,
                "thread": None,
            }

        target_image_id = job["image_id"]
        job_id = job["job_id"]
        stored_name = Path(job["url"]).name

        def _run() -> None:
            self._run_job(
                job_id=job_id,
                image_id=target_image_id,
                stored_name=stored_name,
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                negative_prompt=negative_prompt,
                seed=seed,
                source_image_id=source_image_id,
                source_attachment_id=source_attachment_id,
                strength=strength,
                mask_path=mask_path,
                cancel=cancel,
            )

        thread = threading.Thread(target=_run, name=f"image-{job_id}", daemon=True)
        with self._jobs_lock:
            self._jobs[job_id]["thread"] = thread
        thread.start()
        return job

    def _run_job(
        self,
        *,
        job_id: str,
        image_id: int,
        stored_name: str,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        negative_prompt: str,
        seed: int,
        source_image_id: int | None,
        source_attachment_id: int | None,
        strength: float | None,
        mask_path: str | None,
        cancel: threading.Event,
    ) -> None:
        engine = self.engine
        output_path = os.path.join(self.generated_dir, stored_name)

        def on_progress(step: int, total: int) -> None:
            with self._jobs_lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["progress"] = {"step": step, "total": total}

        try:
            # Wait for the single heavy-job slot; bail out if cancelled meanwhile.
            while not self._run_lock.acquire(timeout=0.5):
                if cancel.is_set():
                    self._set_status(image_id, "cancelled")
                    return
            try:
                if cancel.is_set():
                    self._set_status(image_id, "cancelled")
                    return

                with self._jobs_lock:
                    if job_id in self._jobs:
                        self._jobs[job_id]["running"] = True

                ready, reason = engine.available()
                if not ready:
                    raise ImageEngineError(reason or "The image engine is not ready.")
                if not self.rm.can_spawn_heavy_task():
                    raise ImageEngineError(FRIENDLY_BUSY)

                self._set_status(image_id, "generating")
                self.rm.mark_image_engine_ready()

                try:
                    self._dispatch(
                        engine,
                        prompt=prompt,
                        output_path=output_path,
                        width=width,
                        height=height,
                        steps=steps,
                        negative_prompt=negative_prompt,
                        seed=seed,
                        source_image_id=source_image_id,
                        source_attachment_id=source_attachment_id,
                        strength=strength,
                        mask_path=mask_path,
                        cancel=cancel,
                        on_progress=on_progress,
                    )
                finally:
                    # The engine subprocess is gone; release the RAM it held.
                    self.rm.mark_image_engine_unready()

                if cancel.is_set():
                    _silent_remove(output_path)
                    self._set_status(image_id, "cancelled")
                    return

                thumb = _make_thumbnail(output_path, self.thumbnails_dir)
                row = self.db.get_generated_image(image_id)
                self.db.update_generated_image_status(
                    image_id,
                    status="completed",
                    error_text=None,
                    completed_at=_utcnow(),
                )
                if thumb and row:
                    self._set_thumbnail(image_id, Path(thumb).name)
                logger.info("Image job completed job_id=%s", job_id)
            finally:
                self._run_lock.release()
        except Exception as exc:  # never let a job kill the worker/server
            if cancel.is_set():
                # Cancellation races with subprocess teardown: a killed sd-cli
                # exits non-zero and would otherwise overwrite the cancelled
                # status with a scary "failed" error.
                _silent_remove(output_path)
                self._set_status(image_id, "cancelled")
                logger.info("Image job cancelled job_id=%s", job_id)
            else:
                self._handle_failure(image_id, job_id, exc)
        finally:
            with self._jobs_lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["running"] = False

    def _dispatch(
        self,
        engine,
        *,
        prompt,
        output_path,
        width,
        height,
        steps,
        negative_prompt,
        seed,
        source_image_id,
        source_attachment_id,
        strength,
        mask_path,
        cancel,
        on_progress,
    ) -> str:
        """Route to generate / img2img / inpaint through the ImageEngine API."""
        source_path = self.resolve_source_path(
            source_image_id, source_attachment_id
        )

        if mask_path and source_path:
            try:
                return engine.inpaint(
                    source_path,
                    mask_path,
                    prompt,
                    output_path,
                    strength=strength if strength is not None else 0.55,
                    width=width,
                    height=height,
                    steps=steps,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    cancel=cancel,
                    on_progress=on_progress,
                )
            except UnsupportedOperation:
                raise
        if source_path:
            return engine.img2img(
                source_path,
                prompt,
                output_path,
                strength=strength if strength is not None else 0.55,
                width=width,
                height=height,
                steps=steps,
                negative_prompt=negative_prompt,
                seed=seed,
                cancel=cancel,
                on_progress=on_progress,
            )
        return engine.generate(
            prompt,
            output_path,
            width=width,
            height=height,
            steps=steps,
            negative_prompt=negative_prompt,
            seed=seed,
            cancel=cancel,
            on_progress=on_progress,
        )

    def resolve_source_path(
        self,
        image_id: int | None = None,
        attachment_id: int | None = None,
        generated_dir: str | None = None,
        uploads_dir: str | None = None,
    ) -> str | None:
        """Resolve a source image to a path inside an approved directory."""
        if image_id:
            row = self.db.get_generated_image(image_id)
            if not row:
                return None
            root = os.path.realpath(generated_dir or self.generated_dir)
            candidate = os.path.realpath(os.path.join(root, row["stored_filename"]))
            if candidate.startswith(root + os.sep):
                return candidate if os.path.exists(candidate) else None
            return None
        if attachment_id:
            row = self.db.get_attachment(attachment_id)
            if not row:
                return None
            root = os.path.realpath(
                uploads_dir
                or getattr(self.settings, "uploads_dir_path", None)
                or "data/uploads"
            )
            candidate = os.path.realpath(os.path.join(root, row["stored_filename"]))
            if candidate.startswith(root + os.sep):
                return candidate if os.path.exists(candidate) else None
        return None

    # -- queries -----------------------------------------------------------

    def get_job(self, job_id: str) -> dict | None:
        row = self.db.get_generated_image_by_job(job_id)
        if not row:
            return None
        with self._jobs_lock:
            state = self._jobs.get(job_id) or {}
        progress = state.get("progress")
        status = row["status"]
        # A job still marked queued/generating but unknown to this process was
        # interrupted by a restart: report it as failed instead of hanging.
        if status in ("queued", "generating") and not state:
            status = "failed"
        result = {
            "job_id": job_id,
            "status": status,
            "image_id": row["id"],
            "conversation_id": row["conversation_id"],
            "url": f"/api/generated/{row['stored_filename']}",
            "error_text": row.get("error_text"),
            "progress": progress,
            "prompt": row.get("prompt"),
        }
        return result

    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation. Returns True if the job was still cancellable."""
        with self._jobs_lock:
            state = self._jobs.get(job_id)
        row = self.db.get_generated_image_by_job(job_id)
        if not row:
            return False
        if row["status"] not in ("queued", "generating"):
            return False

        if state is None:
            # Job from a previous process: mark it cancelled directly.
            self._set_status(row["id"], "cancelled")
            return True

        state["cancel"].set()
        if state.get("running"):
            cancel_engine = getattr(self.engine, "cancel", None)
            if callable(cancel_engine):
                try:
                    cancel_engine()
                except Exception:
                    logger.debug("engine cancel failed", exc_info=True)
        self._set_status(row["id"], "cancelled")
        logger.info("Image job cancellation requested job_id=%s", job_id)
        return True

    # -- helpers -----------------------------------------------------------

    def _set_status(
        self, image_id: int, status: str, error_text: str | None = None
    ) -> None:
        completed_at = _utcnow() if status in (
            "completed",
            "failed",
            "cancelled",
        ) else None
        self.db.update_generated_image_status(
            image_id, status=status, error_text=error_text, completed_at=completed_at
        )

    def _set_thumbnail(self, image_id: int, filename: str) -> None:
        conn = self.db._get_connection()
        conn.execute(
            "UPDATE generated_images SET thumbnail_filename=? WHERE id=?",
            (filename, image_id),
        )
        conn.commit()

    def _handle_failure(self, image_id: int, job_id: str, exc: Exception) -> None:
        try:
            if isinstance(exc, GenerationCancelled):
                self._set_status(image_id, "cancelled")
                logger.info("Image job cancelled job_id=%s", job_id)
                return
            if isinstance(exc, UnsupportedOperation):
                self._set_status(image_id, "failed", str(exc))
                logger.warning("Unsupported image operation job_id=%s: %s", job_id, exc)
                return
            if isinstance(exc, ImageEngineError):
                self._set_status(image_id, "failed", str(exc))
                logger.warning("Image job failed job_id=%s: %s", job_id, exc)
                return
            logger.exception("Image job crashed job_id=%s", job_id)
            self._set_status(image_id, "failed", FRIENDLY_FAILURE)
        except Exception:
            # A DB hiccup must never propagate out of the worker thread.
            logger.exception("Could not record failure for job_id=%s", job_id)


def _make_thumbnail(source_path: str, thumbnails_dir: str) -> str | None:
    try:
        from PIL import Image

        if not os.path.exists(source_path):
            return None
        os.makedirs(thumbnails_dir, exist_ok=True)
        name = Path(source_path).stem + ".png"
        target = os.path.join(thumbnails_dir, name)
        with Image.open(source_path) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            im.save(target, format="PNG")
        return target
    except Exception:
        logger.debug("thumbnail creation failed", exc_info=True)
        return None


def _silent_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        logger.debug("could not remove %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Process-wide accessor
# ---------------------------------------------------------------------------

_image_service: ImageService | None = None
_image_service_lock = threading.Lock()
_default_engine = None


def configure_image_service(engine=None) -> None:
    """Test/embedding hook: force an engine (e.g. a fake) for new services.

    Calling with no engine restores automatic detection.
    """
    global _default_engine, _image_service
    _default_engine = engine
    with _image_service_lock:
        _image_service = None


def get_image_service(db=None, settings=None) -> ImageService:
    """Return the shared service so job threads and status polling agree.

    Rebuilt whenever the database identity changes (tests swap DBs).
    """
    global _image_service

    if db is None:
        from app.models.database import get_database as _get_database

        db = _get_database()
    if settings is None:
        from app.config import get_settings as _get_settings

        settings = _get_settings()

    with _image_service_lock:
        # Key on the database path, not object identity: some callers build a
        # temporary Database for the same file, and losing the in-memory job
        # registry mid-generation would break polling and cancellation.
        if (
            _image_service is None
            or getattr(_image_service.db, "_path", None) != getattr(db, "_path", None)
        ):
            from app.core.resource_manager import get_resource_manager

            _image_service = ImageService(
                db=db,
                rm=get_resource_manager(settings=settings),
                generated_dir=settings.generated_dir_path,
                thumbnails_dir=settings.thumbnails_dir_path,
                engine=_default_engine,
                settings=settings,
            )
        return _image_service
