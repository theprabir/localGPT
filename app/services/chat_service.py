import logging

from app.inference.vlm import VLM, VLMModelUnavailable, VLMUnavailable, build_multimodal_message
from app.models.database import Database

logger = logging.getLogger("localgpt.services.chat")


class ChatService:
    def __init__(self, db: Database, vlm: VLM | None = None):
        self.db = db
        self.vlm = vlm or VLM()

    def is_vlm_ready(self) -> bool:
        return self.vlm.is_ready()

    def chat(self, conversation_id: int, content: str, *, attachment_id: int | None = None, image_bytes: bytes | None = None, image_media_type: str = "image/png") -> str:
        if not self.vlm.is_ready():
            raise VLMUnavailable("VLM is not available")

        if attachment_id is not None:
            user_content = f"[Attached image: attachment #{attachment_id}]\n\n{content}"
        else:
            user_content = content

        self.db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_content,
            attachment_id=attachment_id,
        )

        if image_bytes:
            msg = build_multimodal_message(user_content, image_bytes, image_media_type=image_media_type)
            messages = [msg]
        else:
            messages = [{"role": "user", "content": user_content}]

        response = self.vlm.chat(messages)
        assistant_id = self.db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response,
            raw_content=response,
        )
        return response

    def chat_stream(self, conversation_id: int, content: str, *, attachment_id: int | None = None, image_bytes: bytes | None = None, image_media_type: str = "image/png"):
        if not self.vlm.is_ready():
            raise VLMUnavailable("VLM is not available")

        if attachment_id is not None:
            user_content = f"[Attached image: attachment #{attachment_id}]\n\n{content}"
        else:
            user_content = content

        self.db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_content,
            attachment_id=attachment_id,
        )

        if image_bytes:
            msg = build_multimodal_message(user_content, image_bytes, image_media_type=image_media_type)
            messages = [msg]
        else:
            messages = [{"role": "user", "content": user_content}]

        assistant_id = self.db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content="",
            raw_content="",
        )

        full = []
        try:
            for chunk in self.vlm.chat_stream(messages):
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content_chunk = delta.get("content")
                if content_chunk:
                    full.append(content_chunk)
                yield content_chunk or ""
        except VLMModelUnavailable:
            raise
        except VLMUnavailable:
            raise
        except Exception as exc:
            logger.exception("Chat stream error: %s", exc)
            raise VLMUnavailable(str(exc)) from exc

        assembled = "".join(full)
        try:
            self.db.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=assembled,
                raw_content=assembled,
            )
        except Exception:
            logger.exception("Failed to persist assistant message after streaming")
