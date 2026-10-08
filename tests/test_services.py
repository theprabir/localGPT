import os
import tempfile
import pytest
from unittest.mock import MagicMock

from app.services.chat_service import ChatService
from app.services.upload_service import UploadService
from app.services.conversation_service import ConversationService
from app.models.database import Database
from app.inference.vlm import VLM, VLMUnavailable


class TestConversationService:
    def test_create_and_list(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        svc = ConversationService(db)
        try:
            id1 = svc.create("First")
            id2 = svc.create("Second")
            convs = svc.list(limit=10)
            ids = [c["id"] for c in convs]
            assert id1 in ids and id2 in ids
            assert svc.get(id1)["title"] == "First"
            svc.update_title(id1, "Renamed")
            assert svc.get(id1)["title"] == "Renamed"
            assert svc.delete(id2) is True
            assert svc.get(id2) is None
        finally:
            db.close()

    def test_empty_title_fails(self, tmp_path):
        db = Database(str(tmp_path / "db.db"))
        svc = ConversationService(db)
        try:
            with pytest.raises(ValueError):
                svc.create("   ")
        finally:
            db.close()


class TestUploadService:
    def test_store_valid_image(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        uploads = tmp_path / "uploads"
        svc = UploadService(db, str(uploads))

        conv_id = db.create_conversation("Chat")
        result = svc.store(
            conversation_id=conv_id,
            file_bytes=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
            filename="photo.png",
            media_type="image/png",
            max_size_bytes=10 * 1024 * 1024,
        )
        assert result["media_type"] == "image/png"
        assert result["url"].startswith("/uploads/")
        assert os.path.exists(uploads / result["stored_filename"])

    def test_store_too_large(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        uploads = tmp_path / "uploads"
        svc = UploadService(db, str(uploads))

        conv_id = db.create_conversation("Chat")
        with pytest.raises(ValueError, match="too large"):
            svc.store(
                conversation_id=conv_id,
                file_bytes=b"x" * 20,
                filename="photo.png",
                media_type="image/png",
                max_size_bytes=10,
            )


class TestChatService:
    def test_vlm_unavailable_blocks_chat(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        vlm = MagicMock(spec=VLM)
        vlm.is_ready.return_value = False
        svc = ChatService(db, vlm=vlm)
        assert svc.is_vlm_ready() is False
        with pytest.raises(VLMUnavailable):
            svc.chat(1, "Hi")

    def test_stream_raises_when_unavailable(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        vlm = MagicMock(spec=VLM)
        vlm.is_ready.return_value = False
        svc = ChatService(db, vlm=vlm)
        with pytest.raises(VLMUnavailable):
            list(svc.chat_stream(1, "Hi"))
