import logging

from app.models.database import Database

logger = logging.getLogger("localgpt.services.conversation")


class ConversationService:
    def __init__(self, db: Database):
        self.db = db

    def list(self, limit: int = 50, offset: int = 0) -> list[dict]:
        if limit < 1:
            limit = 1
        if limit > 200:
            limit = 200
        return self.db.list_conversations(limit=limit, offset=offset)

    def create(self, title: str, metadata: str | None = None) -> int:
        if not title.strip():
            raise ValueError("Conversation title is required")
        return self.db.create_conversation(title=title.strip(), metadata=metadata)

    def get(self, conversation_id: int) -> dict | None:
        return self.db.get_conversation(conversation_id)

    def delete(self, conversation_id: int) -> bool:
        return self.db.delete_conversation(conversation_id)

    def update_title(self, conversation_id: int, title: str) -> None:
        if not title.strip():
            raise ValueError("Title is required")
        self.db.update_conversation_title(conversation_id, title.strip())
