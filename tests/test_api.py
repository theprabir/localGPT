import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import create_app
from app.config import get_settings
from app.core.logging import setup_logging
from app.models.database import Database
from app.core.resource_manager import ResourceManager
from app.inference.vlm import VLM


def make_test_app():
    settings = get_settings()
    log_file = "/tmp/localgpt-test.log"
    logger = setup_logging(log_file=log_file)
    db = Database(settings=settings)
    rm = ResourceManager(settings=settings)
    app = create_app(settings=settings, logger_instance=logger)
    app.state.db = db
    app.state.rm = rm
    return app, db, rm, logger


@pytest.fixture
def client():
    app, db, rm, logger = make_test_app()
    with TestClient(app) as c:
        yield c
    db.close()


def test_root_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True


def test_settings(client):
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app_name"] == "LocalGPT"
    assert data["vlm_model"] == "smolvlm2-2.2b"


def test_list_conversations_empty(client):
    from app.models.database import Database
    db = Database(settings=client.app.state.settings)
    db.set_setting("conversations_scanned", "true")
    resp = client.get("/api/conversations?offset=9999&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    # Offset beyond existing rows should return an empty list.
    assert data["conversations"] == []


def test_create_and_list_conversation(client):
    resp = client.post("/api/conversations", json={"title": "Test chat"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Test chat"
    assert "id" in data

    conv_id = data["id"]
    resp = client.get(f"/api/conversations/{conv_id}")
    assert resp.status_code == 200
    assert resp.json()["title"] == "Test chat"

    # messages list
    resp = client.get(f"/api/conversations/{conv_id}/messages")
    assert resp.status_code == 200
    assert resp.json()["messages"] == []


def test_delete_conversation(client):
    resp = client.post("/api/conversations", json={"title": "To delete"})
    conv_id = resp.json()["id"]

    resp = client.delete(f"/api/conversations/{conv_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    resp = client.get(f"/api/conversations/{conv_id}")
    assert resp.status_code == 404


def test_chat_stream_without_vlm(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.chat._get_vlm", lambda: MagicMock(ensure_ready=lambda: False)
    )
    resp = client.post(
        "/api/chat/stream",
        json={"content": "Hello"},
    )
    assert resp.status_code == 200
    assert "model_unavailable" in resp.text


def test_upload_invalid_type(client):
    files = {"file": ("test.exe", b"bad", "application/x-executable")}
    data = {"conversation_id": 1}
    resp = client.post("/api/upload", files=files, data=data)
    assert resp.status_code == 400


def test_upload_valid_image(client):
    conv_resp = client.post("/api/conversations", json={"title": "Upload test"})
    conv_id = conv_resp.json()["id"]

    files = {"file": ("photo.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "image/png")}
    data = {"conversation_id": conv_id}
    resp = client.post("/api/upload", files=files, data=data)
    assert resp.status_code == 200
    data = resp.json()
    assert data["media_type"] == "image/png"
    assert data["url"].startswith("/uploads/")


def test_stream_persists_user_message_when_vlm_down(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.chat._get_vlm", lambda: MagicMock(ensure_ready=lambda: False)
    )
    conv_id = client.post("/api/conversations", json={"title": "Persist"}).json()[
        "id"
    ]

    resp = client.post(
        "/api/chat/stream",
        json={"conversation_id": conv_id, "content": "Keep me"},
    )
    assert resp.status_code == 200
    assert "model_unavailable" in resp.text

    msgs = client.get(f"/api/conversations/{conv_id}/messages").json()["messages"]
    assert any(m["role"] == "user" and m["content"] == "Keep me" for m in msgs)


def test_regenerate_drops_stale_answer_without_duplicating_user(
    client, monkeypatch
):
    monkeypatch.setattr(
        "app.api.chat._get_vlm", lambda: MagicMock(ensure_ready=lambda: False)
    )
    conv_id = client.post("/api/conversations", json={"title": "Regen"}).json()[
        "id"
    ]

    # Seed an existing turn.
    db = client.app.state.db
    db.add_message(conversation_id=conv_id, role="user", content="Original question")
    db.add_message(
        conversation_id=conv_id, role="assistant", content="Original answer"
    )

    resp = client.post(
        "/api/chat/stream",
        json={
            "conversation_id": conv_id,
            "content": "Original question",
            "regenerate": True,
        },
    )
    assert resp.status_code == 200
    assert "model_unavailable" in resp.text

    msgs = client.get(f"/api/conversations/{conv_id}/messages").json()["messages"]
    assert len([m for m in msgs if m["role"] == "user"]) == 1
    # Stale answer removed; nothing new written because the VLM is down.
    assert len([m for m in msgs if m["role"] == "assistant"]) == 0


def test_health_reports_cpu_and_memory(client):
    data = client.get("/health").json()
    assert data["ok"] is True
    assert data["cpu"] is not None and "process_percent" in data["cpu"]
    assert data["memory"] is not None and "total_mb" in data["memory"]
