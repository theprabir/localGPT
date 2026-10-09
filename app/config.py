import os
import sys
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class LocalGPTSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOCALGPT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8000

    log_level: str = "INFO"

    # llama-server integration
    llama_server_url: str = "http://127.0.0.1:8080"
    vlm_model: str = "smolvlm2-2.2b"
    vlm_model_path: str = ""
    vlm_max_context_tokens: int = 2048

    # Image generation defaults (Phase 2 configuration surface)
    image_model: str = "sd-turbo"
    max_image_size: int = 512
    image_steps: int = 2

    # Image engine (Phase 2): stable-diffusion.cpp runs as an external process,
    # mirroring llama-server for the VLM (no Python ML framework resident).
    image_engine: str = "sd_cpp"
    sd_cli_path: str = ""  # empty = auto-detect tools/sd.cpp/sd-cli(.exe), then PATH
    image_model_path: str = ""  # empty = auto-detect models/ tree for a GGUF
    image_job_timeout_seconds: int = 3600  # CPU generation is slow on target hardware
    image_threads: int = 0  # 0 = let sd.cpp pick (physical cores)

    # Upload / file handling limits
    max_upload_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    allowed_image_extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".gif",
    )
    allowed_image_mime_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/bmp",
        "image/gif",
    )

    # Data paths (relative to project root by default)
    data_dir: str = "data"
    conversations_dir: str = "conversations"
    uploads_dir: str = "uploads"
    generated_dir: str = "generated"
    thumbnails_dir: str = "thumbnails"
    cache_dir: str = "cache"
    logs_dir: str = "logs"
    database_path: str = "data/localgpt.db"

    # Application metadata
    app_name: str = "LocalGPT"
    app_creator: str = "Prabir kumar Das"

    # llama.cpp runtime reference
    # Phase 1 pins the tested llama-server / llama.cpp release so operators know what
    # LocalGPT was validated against. Update docs when testing newer stable releases.
    llama_cpp_tested_release: str = "llama.cpp main branch builds after 2025-02-10"

    @property
    def conversations_dir_path(self) -> str:
        return os.path.join(self.data_dir, self.conversations_dir)

    @property
    def uploads_dir_path(self) -> str:
        return os.path.join(self.data_dir, self.uploads_dir)

    @property
    def generated_dir_path(self) -> str:
        return os.path.join(self.data_dir, self.generated_dir)

    @property
    def thumbnails_dir_path(self) -> str:
        return os.path.join(self.data_dir, self.thumbnails_dir)

    @property
    def cache_dir_path(self) -> str:
        return os.path.join(self.data_dir, self.cache_dir)

    @property
    def logs_dir_path(self) -> str:
        return os.path.join(self.data_dir, self.logs_dir)

    @property
    def database_path_path(self) -> str:
        # Follow data_dir unless a dedicated LOCALGPT_DATABASE_PATH overrides
        # it, so LOCALGPT_DATA_DIR isolation (tests, portable installs) covers
        # the database file too.
        if self.database_path and self.database_path != "data/localgpt.db":
            return self.database_path
        return os.path.join(self.data_dir, "localgpt.db")


def _is_likely_64_bit() -> bool:
    try:
        return sys.maxsize > 2**32
    except Exception:
        return True


def _cpu_feature_summary() -> dict:
    summary: dict[str, object] = {
        "architecture": "cpu",
        "bits": 64 if _is_likely_64_bit() else 32,
        "explicit_cuda_assumption": False,
        "assumes_avx2": False,
        "assumes_avx512": False,
        "assumes_vulkan": False,
    }

    try:
        import psutil

        mem = psutil.virtual_memory()
        summary["total_memory_mb"] = round(mem.total / (1024**2), 1)
        summary["available_memory_mb"] = round(mem.available / (1024**2), 1)
    except Exception:
        summary["total_memory_mb"] = None
        summary["available_memory_mb"] = None

    return summary


def read_settings_from_environment() -> LocalGPTSettings:
    return LocalGPTSettings()


@lru_cache
def get_settings() -> LocalGPTSettings:
    return read_settings_from_environment()


def detect_cpu_configuration(settings: LocalGPTSettings | None = None) -> dict:
    if settings is None:
        settings = get_settings()
    return _cpu_feature_summary()


if __name__ == "__main__":
    s = get_settings()
    print("LocalGPT configuration:")
    for name in (
        "host",
        "port",
        "log_level",
        "llama_server_url",
        "vlm_model",
        "vlm_model_path",
        "vlm_max_context_tokens",
        "image_model",
        "max_image_size",
        "image_steps",
        "image_engine",
        "sd_cli_path",
        "image_model_path",
        "image_job_timeout_seconds",
        "image_threads",
        "max_upload_size_bytes",
        "app_name",
        "app_creator",
        "llama_cpp_tested_release",
    ):
        print(f"{name}: {getattr(s, name)}")
    print("CPU/feature summary:", detect_cpu_configuration(s))
