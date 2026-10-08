import os
import sys
import importlib

import pytest


def test_settings_defaults():
    from app.config import LocalGPTSettings

    s = LocalGPTSettings()
    assert s.host == "127.0.0.1"
    assert s.port == 8000
    assert s.log_level == "INFO"
    assert s.llama_server_url == "http://127.0.0.1:8080"
    assert s.vlm_model == "smolvlm2-2.2b"
    assert s.max_image_size == 512
    assert s.image_steps == 2
    assert s.app_name == "LocalGPT"
    assert s.app_creator == "Prabir kumar Das"


def test_settings_paths():
    from app.config import LocalGPTSettings

    s = LocalGPTSettings(data_dir="data")
    assert s.conversations_dir_path == os.path.normpath(os.path.join("data", "conversations"))
    assert s.uploads_dir_path == os.path.normpath(os.path.join("data", "uploads"))
    assert s.generated_dir_path == os.path.normpath(os.path.join("data", "generated"))
    assert s.thumbnails_dir_path == os.path.normpath(os.path.join("data", "thumbnails"))
    assert s.cache_dir_path == os.path.normpath(os.path.join("data", "cache"))
    assert s.logs_dir_path == os.path.normpath(os.path.join("data", "logs"))
    assert s.database_path_path == os.path.normpath(os.path.join("data", "localgpt.db")) or s.database_path_path == "data/localgpt.db"


def test_cpu_feature_summary():
    from app.config import detect_cpu_configuration

    summary = detect_cpu_configuration()
    assert summary["architecture"] == "cpu"
    assert summary["explicit_cuda_assumption"] is False
    assert summary["assumes_avx2"] is False


def test_settings_env_override(monkeypatch):
    from app.config import read_settings_from_environment

    monkeypatch.setenv("LOCALGPT_HOST", "0.0.0.0")
    monkeypatch.setenv("LOCALGPT_PORT", "9000")
    monkeypatch.setenv("LOCALGPT_LLAMA_SERVER_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("LOCALGPT_VLM_MODEL", "qwen3-vl-2b")
    monkeypatch.setenv("LOCALGPT_LOG_LEVEL", "DEBUG")

    s = read_settings_from_environment()
    assert s.host == "0.0.0.0"
    assert s.port == 9000
    assert s.llama_server_url == "http://127.0.0.1:9999"
    assert s.vlm_model == "qwen3-vl-2b"
    assert s.log_level == "DEBUG"
