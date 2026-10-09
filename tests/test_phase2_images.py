"""Phase 2 tests: image jobs, unified routing, engine status, generated files.

Inference is mocked (FakeEngine) — no sd-cli, no llama-server, no model
downloads (AGENTS.md §33: automated tests must not require real models).
"""

import json
import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core.logging import setup_logging
from app.core.resource_manager import ResourceManager
from app.inference.image_engine import GenerationCancelled, UnsupportedOperation
from app.main import create_app
from app.models.database import Database
from app.services.image_service import configure_image_service

import io


def _png_bytes():
    """A real 2x2 PNG so the thumbnailer (Pillow) can open it."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (220, 40, 40)).save(buf, format="PNG")
    return buf.getvalue()


PNG_1PX = _png_bytes()


class FakeEngine:
    """Deterministic stand-in for sd-cli; never spawns a process."""

    name = "fake"

    def __init__(self, ready=True, reason=None, gate=None, txt2img=True, fail=None):
        self.ready = ready
        self.reason = reason
        self.gate = gate  # threading.Event: block generate() until released
        self.txt2img = txt2img
        self.fail = fail
        self.calls = []

    def available(self):
        if self.ready:
            return True, None
        return False, self.reason or "The fake image engine is not available."

    def capabilities(self):
        return {"txt2img": self.txt2img, "img2img": True, "inpaint": False}

    def cancel(self):
        pass

    def _finish(self, kind, output_path, cancel):
        self.calls.append(kind)
        if self.gate is not None:
            deadline = time.time() + 10.0
            while not self.gate.wait(0.05):
                if cancel is not None and cancel.is_set():
                    raise GenerationCancelled("cancelled while gated")
                if time.time() > deadline:
                    raise RuntimeError("FakeEngine gate was never released")
        if self.fail is not None:
            raise self.fail
        with open(output_path, "wb") as fh:
            fh.write(PNG_1PX)
        return output_path

    def generate(self, prompt, output_path, **kwargs):
        on_progress = kwargs.get("on_progress")
        if on_progress:
            on_progress(1, 2)
        result = self._finish("generate", output_path, kwargs.get("cancel"))
        if on_progress:
            on_progress(2, 2)
        return result

    def img2img(self, source_path, prompt, output_path, **kwargs):
        return self._finish("img2img", output_path, kwargs.get("cancel"))

    def inpaint(self, source_path, mask_path, prompt, output_path, **kwargs):
        raise UnsupportedOperation("The configured image engine cannot inpaint.")


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    """Every test in this module runs against a FakeEngine.

    can_spawn_heavy_task is stubbed so tests never depend on how much RAM the
    host happens to have free.
    """
    monkeypatch.setattr(
        ResourceManager, "can_spawn_heavy_task", lambda self: True
    )
    engine = FakeEngine()
    configure_image_service(engine)
    yield engine
    configure_image_service(None)


@pytest.fixture
def client():
    settings = get_settings()
    logger = setup_logging(log_file="/tmp/localgpt-test.log")
    db = Database(settings=settings)
    rm = ResourceManager(settings=settings)
    app = create_app(settings=settings, logger_instance=logger)
    app.state.db = db
    app.state.rm = rm
    with TestClient(app) as c:
        yield c
    db.close()


def _conv(client, title="Image chat"):
    resp = client.post("/api/conversations", json={"title": title})
    assert resp.status_code == 200
    return resp.json()["id"]


def _wait_job(client, job_id, timeout=8.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/api/image/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["status"] not in ("queued", "generating"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not settle: {last}")


def _sse_events(text):
    out = []
    for raw in re.findall(r"data: (.+)", text):
        raw = raw.strip()
        if raw == "[DONE]":
            continue
        try:
            out.append(json.loads(raw))
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Engine status exposure
# ---------------------------------------------------------------------------


def test_models_endpoint_exposes_image_engine(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    engine = resp.json()["image_engine"]
    assert engine["ready"] is True
    assert engine["engine"] == "fake"
    assert engine["capabilities"]["txt2img"] is True


def test_system_status_exposes_image_engine(client):
    resp = client.get("/api/system/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["image_engine"]["ready"] is True
    assert data["version"] == "0.1.0-phase2"


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


def test_generate_job_completes_and_file_serves(client, fake_engine):
    conv_id = _conv(client)
    resp = client.post(
        "/api/image/generate",
        json={"conversation_id": conv_id, "prompt": "a red circle"},
    )
    assert resp.status_code == 200, resp.text
    job = resp.json()
    assert job["job_id"].startswith("img_")
    assert job["status"] == "queued"

    final = _wait_job(client, job["job_id"])
    assert final["status"] == "completed", final
    assert final["progress"] is None or final["progress"]["step"] >= 1
    assert fake_engine.calls == ["generate"]

    file_resp = client.get(final["url"])
    assert file_resp.status_code == 200
    assert file_resp.headers["content-type"].startswith("image/")
    assert file_resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_requires_existing_conversation(client):
    resp = client.post(
        "/api/image/generate",
        json={"conversation_id": 999999, "prompt": "nope"},
    )
    assert resp.status_code == 404


def test_generate_unavailable_engine_returns_503(client):
    configure_image_service(
        FakeEngine(ready=False, reason="sd-cli was not found on this machine.")
    )
    conv_id = _conv(client)
    resp = client.post(
        "/api/image/generate",
        json={"conversation_id": conv_id, "prompt": "x"},
    )
    assert resp.status_code == 503
    assert "sd-cli" in resp.json()["detail"]


def test_generate_missing_capability_returns_501(client):
    configure_image_service(FakeEngine(txt2img=False))
    conv_id = _conv(client)
    resp = client.post(
        "/api/image/generate",
        json={"conversation_id": conv_id, "prompt": "x"},
    )
    assert resp.status_code == 501
    assert "txt2img" in resp.json()["detail"]


def test_ownership_source_image_from_other_conversation_rejected(client):
    owner = _conv(client, "owner")
    other = _conv(client, "other")
    resp = client.post(
        "/api/image/generate",
        json={"conversation_id": owner, "prompt": "first"},
    )
    job_id = resp.json()["job_id"]
    final = _wait_job(client, job_id)
    image_id = final["image_id"]

    # img2img against a conversation the image does not belong to -> 403
    resp = client.post(
        "/api/image/generate",
        json={"conversation_id": other, "prompt": "recolor", "image_id": image_id},
    )
    assert resp.status_code == 403

    resp = client.post(
        "/api/image/edit",
        json={"conversation_id": other, "prompt": "recolor", "image_id": image_id},
    )
    assert resp.status_code == 403


def test_job_endpoints_unknown_job_404(client):
    assert client.get("/api/image/jobs/img_nope").status_code == 404
    assert client.post("/api/image/jobs/img_nope/cancel").status_code == 404


def test_cancel_running_job(client, fake_engine):
    gate = threading.Event()
    configure_image_service(FakeEngine(gate=gate))
    conv_id = _conv(client)
    resp = client.post(
        "/api/image/generate",
        json={"conversation_id": conv_id, "prompt": "slow"},
    )
    job_id = resp.json()["job_id"]

    # Wait until the worker is actually inside generate().
    deadline = time.time() + 5.0
    while time.time() < deadline:
        status = client.get(f"/api/image/jobs/{job_id}").json()["status"]
        if status == "generating":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("job never reached 'generating'")

    cancel = client.post(f"/api/image/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    body = cancel.json()
    assert body["cancelled"] is True
    assert body["status"] == "cancelled"

    gate.set()  # let the worker unwind
    final = _wait_job(client, job_id)
    assert final["status"] == "cancelled"
    time.sleep(0.3)  # give the worker thread time to finish cleanly


def test_cancel_finished_job_reports_false(client):
    conv_id = _conv(client)
    resp = client.post(
        "/api/image/generate",
        json={"conversation_id": conv_id, "prompt": "quick"},
    )
    job_id = resp.json()["job_id"]
    final = _wait_job(client, job_id)
    assert final["status"] == "completed"

    cancel = client.post(f"/api/image/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["cancelled"] is False
    assert cancel.json()["status"] == "completed"


def test_failed_engine_run_reports_friendly_error(client):
    configure_image_service(FakeEngine(fail=RuntimeError("boom trace here")))
    conv_id = _conv(client)
    resp = client.post(
        "/api/image/generate",
        json={"conversation_id": conv_id, "prompt": "boom"},
    )
    job_id = resp.json()["job_id"]
    final = _wait_job(client, job_id)
    assert final["status"] == "failed"
    # The user-facing text must be friendly: no raw traceback/exception repr.
    assert "boom" not in (final["error_text"] or "")
    assert "Image generation failed" in (final["error_text"] or "")


def test_generated_images_listing_and_traversal_guard(client):
    conv_id = _conv(client)
    resp = client.post(
        "/api/image/generate",
        json={"conversation_id": conv_id, "prompt": "listed"},
    )
    final = _wait_job(client, resp.json()["job_id"])
    assert final["status"] == "completed"

    listing = client.get(f"/api/conversations/{conv_id}/images")
    assert listing.status_code == 200
    images = listing.json()["images"]
    assert len(images) == 1
    assert images[0]["status"] == "completed"
    assert images[0]["prompt"] == "listed"

    # Path traversal must never resolve outside the approved directories.
    for bad in (
        "/api/generated/..%2F..%2Flocalgpt.db",
        "/api/generated/%2e%2e%2fconfig.py",
    ):
        resp = client.get(bad)
        assert resp.status_code in (400, 404), (bad, resp.status_code)


# ---------------------------------------------------------------------------
# Unified routing through chat
# ---------------------------------------------------------------------------


def test_chat_stream_routes_image_generation(client, fake_engine):
    conv_id = _conv(client)
    resp = client.post(
        "/api/chat/stream",
        json={"conversation_id": conv_id, "content": "Draw a red circle"},
    )
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    image_events = [e for e in events if e.get("event") == "image_job"]
    assert image_events, resp.text
    ev = image_events[0]
    assert ev["status"] == "queued"
    assert ev["job_id"].startswith("img_")
    assert "Draw a red circle" in ev["message"]
    assert any(e.get("event") == "done" for e in events)

    final = _wait_job(client, ev["job_id"])
    assert final["status"] == "completed"
    assert fake_engine.calls == ["generate"]

    # The routing decision is persisted as an assistant message.
    msgs = client.get(f"/api/conversations/{conv_id}/messages").json()["messages"]
    assert any(
        m["role"] == "assistant" and m["content"].startswith("Generating image:")
        for m in msgs
    )


def test_chat_non_stream_returns_image_job_id(client):
    conv_id = _conv(client)
    resp = client.post(
        "/api/chat",
        json={"conversation_id": conv_id, "content": "Generate a logo of a fox"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["image_job_id"]
    assert data["content"].startswith("Generating image:")
    _wait_job(client, data["image_job_id"])


def test_chat_stream_image_edit_without_attachment_explains(client):
    conv_id = _conv(client)
    resp = client.post(
        "/api/chat/stream",
        json={"conversation_id": conv_id, "content": "Change the color of the car"},
    )
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    image_events = [e for e in events if e.get("event") == "image_job"]
    assert image_events
    assert image_events[0]["status"] == "unavailable"
    assert image_events[0]["job_id"] is None
    assert "Attach the image" in image_events[0]["message"]


def test_chat_stream_image_intent_when_engine_down(client):
    configure_image_service(
        FakeEngine(ready=False, reason="Image model missing.")
    )
    conv_id = _conv(client)
    resp = client.post(
        "/api/chat/stream",
        json={"conversation_id": conv_id, "content": "Draw a castle"},
    )
    events = _sse_events(resp.text)
    image_events = [e for e in events if e.get("event") == "image_job"]
    assert image_events
    assert image_events[0]["status"] == "unavailable"
    assert "Image model missing." in image_events[0]["message"]


def test_chat_plain_text_still_goes_to_vlm_path(client, monkeypatch):
    """Non-image intents must NOT be diverted to the image engine."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        "app.api.chat._get_vlm", lambda: MagicMock(ensure_ready=lambda: False)
    )
    conv_id = _conv(client)
    resp = client.post(
        "/api/chat/stream",
        json={"conversation_id": conv_id, "content": "Explain quantum computing"},
    )
    events = _sse_events(resp.text)
    assert not any(e.get("event") == "image_job" for e in events)
    assert any(e.get("event") == "error" for e in events)


def test_regenerate_is_never_routed_to_image_engine(
    client, fake_engine, monkeypatch
):
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        "app.api.chat._get_vlm", lambda: MagicMock(ensure_ready=lambda: False)
    )
    conv_id = _conv(client)
    resp1 = client.post(
        "/api/chat/stream",
        json={"conversation_id": conv_id, "content": "Draw a red circle"},
    )
    routed = [e for e in _sse_events(resp1.text) if e.get("event") == "image_job"]
    assert routed and routed[0]["status"] == "queued"
    # Let the worker finish so the engine call list is deterministic.
    final = _wait_job(client, routed[0]["job_id"])
    assert final["status"] == "completed"
    # Regenerate flag must bypass image routing even for image-like text.
    resp = client.post(
        "/api/chat/stream",
        json={"conversation_id": conv_id, "content": "Draw a red circle", "regenerate": True},
    )
    events = _sse_events(resp.text)
    assert not any(e.get("event") == "image_job" for e in events)
    assert fake_engine.calls == ["generate"]  # only the first turn routed


# ---------------------------------------------------------------------------
# Service-level safety
# ---------------------------------------------------------------------------


def test_resolve_source_path_rejects_traversal(client, fake_engine):
    from app.services.image_service import get_image_service
    from app.config import get_settings as _gs

    settings = _gs()
    svc = get_image_service(db=client.app.state.db, settings=settings)
    # A generated-image row whose stored_filename escapes the root.
    assert svc.resolve_source_path() is None
    assert svc.resolve_source_path(image_id=424242) is None
    assert svc.resolve_source_path(attachment_id=424242) is None


def test_unsupported_inpaint_never_faked(client):
    engine = FakeEngine()
    from app.inference.image_engine import ImageEngineError  # noqa: F401

    with pytest.raises(UnsupportedOperation):
        engine.inpaint("/tmp/src.png", "/tmp/mask.png", "p", "/tmp/out.png")
