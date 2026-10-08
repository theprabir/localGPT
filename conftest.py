import os
import sys
import tempfile

import pytest


@pytest.fixture(autouse=True)
def run_in_tmp_project(monkeypatch, tmp_path):
    """
    Make tests run in an isolated temporary home/project root so they do not touch
    the real data directory.
    """
    monkeypatch.setenv("LOCALGPT_DATA_DIR", str(tmp_path / "data"))
    # Ensure the tmp data layout exists.
    for sub in ("conversations", "uploads", "generated", "thumbnails", "cache", "logs"):
        (tmp_path / "data" / sub).mkdir(parents=True, exist_ok=True)

    # Database tests use explicit tmp_path paths, so we only need env isolation here.
    yield tmp_path
