import logging
import os
import sys
from typing import Optional

from app.config import get_settings


def _build_log_format() -> str:
    return "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def _ensure_directories(settings) -> None:
    for directory in (
        settings.logs_dir_path,
        settings.data_dir,
    ):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as exc:
            print(f"WARNING: could not create directory {directory}: {exc}", file=sys.stderr)


def _configure_root(level: int) -> None:
    handler: logging.Handler
    if sys.stdout.isatty():
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_build_log_format()))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []
    root.addHandler(handler)


def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    include_timestamp_utc: bool = False,
) -> logging.Logger:
    if level is None:
        settings = get_settings()
        level_name = getattr(settings, "log_level", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

    _ensure_directories(get_settings())

    _configure_root(level)

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(_build_log_format()))
            logging.getLogger().addHandler(file_handler)
        except Exception as exc:
            logging.getLogger().warning("Could not open log file %s: %s", log_file, exc)

    logging.Formatter.converter = (
        lambda *args: __import__("time").gmtime()
        if include_timestamp_utc
        else __import__("time").localtime()
    )

    return logging.getLogger("localgpt")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_startup(logger: logging.Logger) -> None:
    settings = get_settings()
    cpu = get_settings().__dict__.get("_cpu_feature_summary", None)

    logger.info("LocalGPT starting")
    logger.info("app_name=%s", settings.app_name)
    logger.info("app_creator=%s", settings.app_creator)
    logger.info("log_level=%s", settings.log_level)
    logger.info("host=%s port=%s", settings.host, settings.port)
    logger.info("llama_server_url=%s", settings.llama_server_url)
    logger.info("vlm_model=%s", settings.vlm_model)
    logger.info("vlm_model_path=%s", settings.vlm_model_path or "(unset)")
    logger.info("vlm_max_context_tokens=%s", settings.vlm_max_context_tokens)
    logger.info("image_model=%s", settings.image_model)
    logger.info("max_image_size=%s", settings.max_image_size)
    logger.info("image_steps=%s", settings.image_steps)
    logger.info("max_upload_size_bytes=%s", settings.max_upload_size_bytes)
    logger.info("cpu_config=%s", detect_cpu_configuration(settings))
    logger.info("data_dir=%s", settings.data_dir)


def detect_cpu_configuration(settings) -> dict:
    from app.config import detect_cpu_configuration as detect

    return detect(settings)


def create_log_file_path(settings) -> str:
    return os.path.join(settings.logs_dir_path, "localgpt.log")
