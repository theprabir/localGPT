import os
import tempfile
import pytest

from app.models.database import Database


class TestDatabaseConversations:
    def test_create_and_list(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        try:
            id1 = db.create_conversation("First")
            id2 = db.create_conversation("Second")

            convs = db.list_conversations(limit=10)
            ids = [c["id"] for c in convs]
            assert id2 in ids
            assert id1 in ids

            conv = db.get_conversation(id1)
            assert conv["title"] == "First"

            db.update_conversation_title(id1, "Renamed")
            conv = db.get_conversation(id1)
            assert conv["title"] == "Renamed"

            assert db.delete_conversation(id2) is True
            assert db.get_conversation(id2) is None
        finally:
            db.close()

    def test_metadata_roundtrip(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        try:
            id1 = db.create_conversation("Test", metadata='{"foo":"bar"}')
            db.update_conversation_metadata(id1, '{"foo":"baz"}')
            conv = db.get_conversation(id1)
            assert conv["metadata"] == '{"foo":"baz"}'
        finally:
            db.close()


class TestDatabaseMessages:
    def test_add_and_list(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        try:
            conv_id = db.create_conversation("Chat")
            m1 = db.add_message(conv_id, "user", "Hello")
            m2 = db.add_message(conv_id, "assistant", "Hi")

            messages = db.list_messages(conv_id)
            ids = [m["id"] for m in messages]
            assert m1 in ids
            assert m2 in ids

            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
        finally:
            db.close()

    def test_message_with_attachment_link(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        try:
            conv_id = db.create_conversation("Chat")
            att_id = db.add_attachment(
                conversation_id=conv_id,
                original_filename="photo.png",
                stored_filename="uuid.png",
                media_type="image/png",
                size_bytes=100,
            )
            msg_id = db.add_message(conv_id, "user", "See this", attachment_id=att_id)
            msg = db.get_message(msg_id)
            assert msg["attachment_id"] == att_id
        finally:
            db.close()


class TestDatabaseAttachments:
    def test_add_and_get(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        try:
            conv_id = db.create_conversation("Chat")
            att_id = db.add_attachment(
                conversation_id=conv_id,
                original_filename="photo.png",
                stored_filename="uuid.png",
                media_type="image/png",
                size_bytes=1234,
                width=512,
                height=512,
            )
            att = db.get_attachment(att_id)
            assert att["original_filename"] == "photo.png"
            assert att["media_type"] == "image/png"
            assert att["size_bytes"] == 1234
            assert att["width"] == 512
            assert att["height"] == 512
        finally:
            db.close()


class TestDatabaseGeneratedImages:
    def test_add_and_update_status(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        try:
            conv_id = db.create_conversation("Chat")
            img_id = db.add_generated_image(
                conversation_id=conv_id,
                job_id="img_abc",
                prompt="A tree",
                stored_filename="uuid.png",
                thumbnail_filename=None,
                media_type="image/png",
                width=512,
                height=512,
                status="queued",
            )
            row = db.get_generated_image(img_id)
            assert row["status"] == "queued"

            db.update_generated_image_status(img_id, status="completed", completed_at="2026-01-01T00:00:00Z")
            row = db.get_generated_image(img_id)
            assert row["status"] == "completed"
            assert row["completed_at"] == "2026-01-01T00:00:00Z"
        finally:
            db.close()


class TestDatabaseSettings:
    def test_set_and_get(self, tmp_path):
        db_path = tmp_path / "localgpt.db"
        db = Database(str(db_path))
        try:
            db.set_setting("theme", "dark")
            assert db.get_setting("theme") == "dark"
            assert db.get_setting("missing") is None
        finally:
            db.close()
