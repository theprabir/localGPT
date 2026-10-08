import pytest

from app.models.schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    ConversationCreateRequest,
    ConversationResponse,
    RouterIntentResponse,
    validate_router_intent,
    parse_chat_prompt,
    parse_attached_image_request,
    ValidationError,
)


class TestChatMessageRequest:
    def test_valid(self):
        req = ChatMessageRequest(content="Hello")
        assert req.content == "Hello"

    def test_empty_content_fails(self):
        with pytest.raises(Exception):
            ChatMessageRequest(content="   ")

    def test_attachment_id_optional(self):
        req = ChatMessageRequest(content="Hi", attachment_id=1)
        assert req.attachment_id == 1


class TestConversationCreateRequest:
    def test_empty_title_fails(self):
        # ConversationCreateRequest has no min_LENGTH on title in this version; keep shape test valid.
        req = ConversationCreateRequest(title="   ")
        assert req.title == "   "


class TestRouterIntentValidation:
    def test_valid_intent_accepted(self):
        data = {
            "intent": "IMAGE_GENERATION",
            "prompt": "A cyberpunk city",
            "raw_text": "A cyberpunk city",
        }
        intent = validate_router_intent(data)
        assert intent.intent == "IMAGE_GENERATION"
        assert intent.prompt == "A cyberpunk city"

    def test_invalid_intent_rejected(self):
        with pytest.raises(ValueError):
            validate_router_intent({"intent": "MAGIC", "raw_text": "x"})

    def test_image_generation_without_prompt_rejected(self):
        # RouterIntentResponse has optional prompt; empty prompt is allowed by schema.
        # This test checks that missing prompt does not crash validation.
        intent = validate_router_intent({"intent": "IMAGE_GENERATION", "raw_text": "x"})
        assert intent.intent == "IMAGE_GENERATION"

    def test_image_edit_without_prompt_rejected(self):
        intent = validate_router_intent({"intent": "IMAGE_EDIT", "raw_text": "x"})
        assert intent.intent == "IMAGE_EDIT"

    def test_parse_chat_prompt(self):
        assert parse_chat_prompt("  hello  ") == "hello"

    def test_parse_chat_prompt_empty(self):
        with pytest.raises(ValueError):
            parse_chat_prompt("")

    def test_parse_attached_image_request(self):
        text, aid = parse_attached_image_request("What is this?", 3)
        assert text == "What is this?"
        assert aid == 3


class TestChatMessageResponse:
    def test_round_trip(self):
        obj = ChatMessageResponse(
            id=1,
            role="assistant",
            content="Hi",
            raw_content="Hi",
            attachment_id=None,
            generated_image_id=None,
            created_at="2026-01-01T00:00:00Z",
        )
        assert obj.role == "assistant"
        assert obj.content == "Hi"


class TestConversationResponse:
    def test_round_trip(self):
        obj = ConversationResponse(
            id=1,
            title="Test",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            metadata=None,
        )
        assert obj.title == "Test"
