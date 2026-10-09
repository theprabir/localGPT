import logging
import time

from fastapi import APIRouter, Depends

from app.config import get_settings
from app.core.logging import get_logger, create_log_file_path
from app.core.resource_manager import get_resource_manager
from app.inference.vlm import VLM, _model_friendly_name
from app.models.database import get_database
from app.services.image_service import get_image_service

logger = get_logger("localgpt.api.system")

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health(
    settings=Depends(get_settings),
    rm=Depends(get_resource_manager),
):
    vlm = VLM()
    vlm_ok = vlm.health()
    return {
        "ok": True,
        "vlm": vlm.describe_status(),
        "cpu": rm.get_cpu_usage() if hasattr(rm, "get_cpu_usage") else None,
        "memory": rm.get_system_memory_usage() if hasattr(rm, "get_system_memory_usage") else None,
    }


@router.get("/settings")
def get_settings_endpoint(
    settings=Depends(get_settings),
):
    return {
        "host": settings.host,
        "port": settings.port,
        "log_level": settings.log_level,
        "llama_server_url": settings.llama_server_url,
        "vlm_model": settings.vlm_model,
        "vlm_model_path": settings.vlm_model_path,
        "vlm_max_context_tokens": settings.vlm_max_context_tokens,
        "image_model": settings.image_model,
        "max_image_size": settings.max_image_size,
        "image_steps": settings.image_steps,
        "max_upload_size_bytes": settings.max_upload_size_bytes,
        "app_name": settings.app_name,
        "app_creator": settings.app_creator,
    }


@router.get("/models")
def list_models(
    settings=Depends(get_settings),
    rm=Depends(get_resource_manager),
    db=Depends(get_database),
):
    vlm = VLM()
    image_service = get_image_service(db=db, settings=settings)
    return {
        "supported": ["smolvlm2-2.2b", "qwen3-vl-2b"],
        "current": settings.vlm_model,
        "model_friendly_name": _model_friendly_name(settings.vlm_model),
        "image_model": settings.image_model,
        "vlm_status": vlm.describe_status(),
        "image_engine": image_service.engine_status(),
        "resource_snapshot": rm.snapshot(),
    }


@router.get("/system/status")
def system_status(
    settings=Depends(get_settings),
    rm=Depends(get_resource_manager),
    db=Depends(get_database),
):
    vlm = VLM()
    image_service = get_image_service(db=db, settings=settings)
    return {
        "running": True,
        "version": "0.1.0-phase2",
        "llama_server_reachable": vlm.client.health() if hasattr(vlm, "client") else None,
        "vlm_model": settings.vlm_model,
        "image_engine": image_service.engine_status(),
        "memory_mb": rm.get_memory_usage().get("rss_mb"),
        "cpu_percent": rm.get_cpu_usage().get("process_percent"),
        "resource_snapshot": rm.snapshot(),
        "log_file": create_log_file_path(settings),
    }
