import logging
import os
import threading
import time
from typing import Any, Callable

from app.config import get_settings

logger = logging.getLogger("localgpt.core.resource_manager")


class ResourceManager:
    """
    Central visibility/audit manager for LocalGPT's heavy components.

    This is deliberately lightweight. It does not eagerly unload models; it tracks
    state, provides memory/CPU snapshots, and exposes hooks for lifecycle decisions.

    Example interface:
        ResourceManager
        ├── ensure_vlm_available()
        ├── ensure_image_engine_available()
        ├── release_image_engine()
        ├── release_vlm()
        ├── get_memory_usage()
        └── get_cpu_usage()
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._vlm_ready = False
        self._image_engine_ready = False
        self._vlm_ready_timestamp: float | None = None
        self._image_engine_ready_timestamp: float | None = None
        self._lock = threading.Lock()

    def mark_vlm_ready(self) -> None:
        with self._lock:
            self._vlm_ready = True
            self._vlm_ready_timestamp = time.monotonic()
        logger.info("Resource manager: VLM marked ready")

    def mark_vlm_unready(self) -> None:
        with self._lock:
            self._vlm_ready = False
            self._vlm_ready_timestamp = None
        logger.info("Resource manager: VLM marked unready")

    def mark_image_engine_ready(self) -> None:
        with self._lock:
            self._image_engine_ready = True
            self._image_engine_ready_timestamp = time.monotonic()
        logger.info("Resource manager: image engine marked ready")

    def mark_image_engine_unready(self) -> None:
        with self._lock:
            self._image_engine_ready = False
            self._image_engine_ready_timestamp = None
        logger.info("Resource manager: image engine marked unready")

    def ensure_vlm_available(self) -> bool:
        with self._lock:
            return self._vlm_ready

    def ensure_image_engine_available(self) -> bool:
        with self._lock:
            return self._image_engine_ready

    def release_vlm(self) -> None:
        self.mark_vlm_unready()

    def release_image_engine(self) -> None:
        self.mark_image_engine_unready()

    def get_memory_usage(self) -> dict:
        try:
            import psutil

            process = psutil.Process(os.getpid())
            mem = process.memory_info()
            return {
                "rss_mb": round(mem.rss / (1024**2), 2),
                "vms_mb": round(mem.vms / (1024**2), 2),
                "shared_mb": round(getattr(mem, "shared", 0) / (1024**2), 2),
            }
        except Exception as exc:
            logger.debug("Resource manager: memory snapshot failed: %s", exc)
            return {"rss_mb": None, "vms_mb": None, "shared_mb": None}

    def get_system_memory_usage(self) -> dict:
        try:
            import psutil

            mem = psutil.virtual_memory()
            return {
                "total_mb": round(mem.total / (1024**2), 2),
                "available_mb": round(mem.available / (1024**2), 2),
                "used_percent": round(mem.percent, 1),
            }
        except Exception as exc:
            logger.debug("Resource manager: system memory snapshot failed: %s", exc)
            return {"total_mb": None, "available_mb": None, "used_percent": None}

    def get_cpu_usage(self) -> dict:
        try:
            import psutil

            process = psutil.Process(os.getpid())
            cpu = process.cpu_percent(interval=0.1)
            system_cpu = psutil.cpu_percent(interval=0.1)
            return {
                "process_percent": round(cpu, 1),
                "system_percent": round(system_cpu, 1),
                "logical_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
            }
        except Exception as exc:
            logger.debug("Resource manager: cpu snapshot failed: %s", exc)
            return {"process_percent": None, "system_percent": None, "logical_cores": None, "physical_cores": None}

    def snapshot(self) -> dict:
        return {
            "vlm_ready": self.ensure_vlm_available(),
            "image_engine_ready": self.ensure_image_engine_available(),
            "memory": self.get_memory_usage(),
            "system_memory": self.get_system_memory_usage(),
            "cpu": self.get_cpu_usage(),
            "settings": {
                "max_image_size": self._settings.max_image_size,
                "image_steps": self._settings.image_steps,
                "max_upload_size_bytes": self._settings.max_upload_size_bytes,
            },
        }

    def can_spawn_heavy_task(self) -> bool:
        sys_mem = self.get_system_memory_usage()
        avail = sys_mem.get("available_mb")
        if avail is None:
            return False
        settings = self._settings
        # Very conservative guard for the 10 GB target machine.
        return avail > 1500


_singleton: "ResourceManager | None" = None
_singleton_lock = threading.Lock()


def get_resource_manager(settings=None) -> ResourceManager:
    """Process-wide resource manager.

    Lifecycle flags (VLM / image engine ready) are meaningless if every
    dependency injection builds a fresh instance, so callers share one object.
    """
    global _singleton
    with _singleton_lock:
        if _singleton is None or (
            settings is not None and _singleton._settings is not settings
        ):
            _singleton = ResourceManager(settings=settings)
        return _singleton
