import logging
import os
import sqlite3
import threading
from typing import Any, Sequence

from app.config import get_settings


class Database:
    def __init__(self, path: str | None = None, settings=None):
        if path is None:
            if settings is None:
                settings = get_settings()
            path = settings.database_path_path

        self._path = os.path.abspath(path)
        self._local = threading.local()
        self._initialize()

    def _initialize(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate(conn)
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        thread = getattr(self._local, "connection", None)
        if thread is None:
            thread = sqlite3.connect(self._path, check_same_thread=False)
            thread.row_factory = sqlite3.Row
            thread.execute("PRAGMA foreign_keys=ON")
            self._local.connection = thread
        return thread

    def close(self) -> None:
        thread = getattr(self._local, "connection", None)
        if thread is not None:
            try:
                thread.close()
            except Exception:
                pass
            self._local.connection = None

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                raw_content TEXT,
                attachment_id INTEGER,
                generated_image_id INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS generated_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                job_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                thumbnail_filename TEXT,
                media_type TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                status TEXT NOT NULL,
                error_text TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_attachments_conversation ON attachments(conversation_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_generated_images_conversation ON generated_images(conversation_id)
        """
        )
        conn.commit()

    # ---- Settings ----

    def set_setting(self, key: str, value: str) -> None:
        conn = self._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    def get_setting(self, key: str) -> str | None:
        conn = self._get_connection()
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    # ---- Conversations ----

    def create_conversation(self, title: str, metadata: str | None = None) -> int:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        cur = conn.execute(
            "INSERT INTO conversations(title, created_at, updated_at, metadata) VALUES (?, ?, ?, ?)",
            (title, now, now, metadata),
        )
        conn.commit()
        return cur.lastrowid

    def list_conversations(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at, metadata FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: int) -> dict[str, Any] | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT id, title, created_at, updated_at, metadata FROM conversations WHERE id=?",
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_conversation_title(self, conversation_id: int, title: str) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        conn.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title, now, conversation_id),
        )
        conn.commit()

    def update_conversation_metadata(self, conversation_id: int, metadata: str) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        conn.execute(
            "UPDATE conversations SET metadata=?, updated_at=? WHERE id=?",
            (metadata, now, conversation_id),
        )
        conn.commit()

    def delete_conversation(self, conversation_id: int) -> bool:
        conn = self._get_connection()
        cur = conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        conn.commit()
        return cur.rowcount > 0

    # ---- Messages ----

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        raw_content: str | None = None,
        attachment_id: int | None = None,
        generated_image_id: int | None = None,
    ) -> int:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        cur = conn.execute(
            "INSERT INTO messages(conversation_id, role, content, raw_content, attachment_id, generated_image_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, raw_content, attachment_id, generated_image_id, now),
        )
        conn.commit()
        return cur.lastrowid

    def list_messages(self, conversation_id: int) -> list[dict[str, Any]]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT id, conversation_id, role, content, raw_content, attachment_id, generated_image_id, created_at FROM messages WHERE conversation_id=? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_message(self, message_id: int) -> dict[str, Any] | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT id, conversation_id, role, content, raw_content, attachment_id, generated_image_id, created_at FROM messages WHERE id=?",
            (message_id,),
        ).fetchone()
        return dict(row) if row else None

    def delete_message(self, message_id: int) -> bool:
        conn = self._get_connection()
        cur = conn.execute("DELETE FROM messages WHERE id=?", (message_id,))
        conn.commit()
        return cur.rowcount > 0

    def get_last_message(self, conversation_id: int) -> dict[str, Any] | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT id, conversation_id, role, content, raw_content, attachment_id, generated_image_id, created_at "
            "FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_message_content(
        self, message_id: int, content: str, raw_content: str | None = None
    ) -> bool:
        conn = self._get_connection()
        cur = conn.execute(
            "UPDATE messages SET content=?, raw_content=? WHERE id=?",
            (content, raw_content if raw_content is not None else content, message_id),
        )
        conn.commit()
        return cur.rowcount > 0

    # ---- Attachments ----

    def add_attachment(
        self,
        conversation_id: int,
        original_filename: str,
        stored_filename: str,
        media_type: str,
        size_bytes: int,
        width: int | None = None,
        height: int | None = None,
    ) -> int:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        cur = conn.execute(
            "INSERT INTO attachments(conversation_id, original_filename, stored_filename, media_type, size_bytes, width, height, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, original_filename, stored_filename, media_type, size_bytes, width, height, now),
        )
        conn.commit()
        return cur.lastrowid

    def get_attachment(self, attachment_id: int) -> dict[str, Any] | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT id, conversation_id, original_filename, stored_filename, media_type, size_bytes, width, height, created_at FROM attachments WHERE id=?",
            (attachment_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_attachments_for_conversation(self, conversation_id: int) -> list[dict[str, Any]]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT id, conversation_id, original_filename, stored_filename, media_type, size_bytes, width, height, created_at FROM attachments WHERE conversation_id=? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ---- Generated images ----

    def add_generated_image(
        self,
        conversation_id: int,
        job_id: str,
        prompt: str,
        stored_filename: str,
        thumbnail_filename: str | None,
        media_type: str,
        width: int,
        height: int,
        status: str,
        error_text: str | None = None,
    ) -> int:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        cur = conn.execute(
            "INSERT INTO generated_images(conversation_id, job_id, prompt, stored_filename, thumbnail_filename, media_type, width, height, status, error_text, created_at, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, job_id, prompt, stored_filename, thumbnail_filename, media_type, width, height, status, error_text, now, None),
        )
        conn.commit()
        return cur.lastrowid

    def update_generated_image_status(
        self,
        image_id: int,
        status: str,
        error_text: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        conn = self._get_connection()
        conn.execute(
            "UPDATE generated_images SET status=?, error_text=?, completed_at=? WHERE id=?",
            (status, error_text, completed_at, image_id),
        )
        conn.commit()

    def get_generated_image(self, image_id: int) -> dict[str, Any] | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT id, conversation_id, job_id, prompt, stored_filename, thumbnail_filename, media_type, width, height, status, error_text, created_at, completed_at FROM generated_images WHERE id=?",
            (image_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_generated_image_by_job(self, job_id: str) -> dict[str, Any] | None:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT id, conversation_id, job_id, prompt, stored_filename, thumbnail_filename, media_type, width, height, status, error_text, created_at, completed_at FROM generated_images WHERE job_id=?",
            (job_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_generated_images_for_conversation(self, conversation_id: int) -> list[dict[str, Any]]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT id, conversation_id, job_id, prompt, stored_filename, thumbnail_filename, media_type, width, height, status, error_text, created_at, completed_at FROM generated_images WHERE conversation_id=? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ---- Utility ----

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Cursor:
        conn = self._get_connection()
        return conn.execute(sql, parameters)

    def commit(self) -> None:
        conn = self._get_connection()
        conn.commit()


def get_database(settings=None) -> Database:
    return Database(settings=settings)
