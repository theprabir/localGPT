import pytest
from unittest.mock import MagicMock

from app.inference.llama_server import (
    LlamaServerClient,
    translate_llama_messages,
    _parse_sse_chunk,
)
from app.inference.vlm import (
    VLM,
    _extract_assistant_text,
    _model_friendly_name,
    build_multimodal_message,
    VLMUnavailable,
    VLMModelUnavailable,
)


class TestTranslateLlamaMessages:
    def test_identity_preservation(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = translate_llama_messages(messages)
        assert result == messages

    def test_none_content_becomes_empty(self):
        result = translate_llama_messages([{"role": "user", "content": None}])
        assert result[0]["content"] == ""


class TestParseSseChunk:
    def test_chunk_with_prefix(self):
        assert _parse_sse_chunk("data: {\"foo\":\"bar\"}") == {"foo": "bar"}

    def test_chunk_without_prefix(self):
        # The caller strips the 'data: ' prefix before parsing, so a bare JSON
        # payload must parse as-is — JSON contains colons of its own and must
        # never be split on ':'.
        assert _parse_sse_chunk('{"foo":"bar"}') == {"foo": "bar"}

    def test_chunk_payload_with_inner_colons(self):
        # Regression: the old parser split on the first ':' and corrupted the
        # JSON payload of every real streaming chunk.
        chunk = '{"choices": [{"delta": {"content": "a: b, c"}}]}'
        assert _parse_sse_chunk("data: " + chunk) == {
            "choices": [{"delta": {"content": "a: b, c"}}]
        }
        assert _parse_sse_chunk(chunk) == {
            "choices": [{"delta": {"content": "a: b, c"}}]
        }

    def test_chunk_done_marker_raises(self):
        with pytest.raises(Exception):
            _parse_sse_chunk("[DONE]")


class TestExtractAssistantText:
    def test_normal(self):
        response = {"choices": [{"message": {"content": "Hello"}}]}
        assert _extract_assistant_text(response) == "Hello"

    def test_empty_choices(self):
        assert _extract_assistant_text({}) == ""

    def test_none_content(self):
        assert _extract_assistant_text({"choices": [{"message": {"content": None}}]}) == ""


class TestModelFriendlyName:
    def test_smolvlm(self):
        assert _model_friendly_name("smolvlm2-2.2b") == "SmolVLM2 2.2B"

    def test_qwen(self):
        assert _model_friendly_name("qwen3-vl-2b") == "Qwen3-VL 2B"

    def test_unknown(self):
        assert _model_friendly_name("unknown-model") == "unknown-model"


class TestVLMHealth:
    def test_unhealthy_client(self, monkeypatch):
        client = MagicMock()
        client.health.return_value = False
        vlm = VLM(client=client)
        assert vlm.health() is False

    def test_model_unavailable_raised(self, monkeypatch):
        client = MagicMock()
        client.health.return_value = True
        client.chat_completions.side_effect = Exception("nope")
        vlm = VLM(client=client)
        with pytest.raises(VLMUnavailable):
            vlm.chat([{"role": "user", "content": "hi"}])


class TestVLMStream:
    def test_stream_adapter_called(self, monkeypatch):
        client = MagicMock()
        client.health.return_value = True
        client.chat_completions_stream.return_value = iter([{"choices": [{"delta": {"content": "a"}}]}])
        vlm = VLM(client=client)
        chunks = list(vlm.chat_stream([{"role": "user", "content": "hi"}]))
        assert chunks[0]["choices"][0]["delta"]["content"] == "a"


class TestBuildMultimodalMessage:
    def test_image_message(self, monkeypatch):
        from PIL import Image as PILImage
        from io import BytesIO
        real = PILImage.open
        def fake_open(*args, **kwargs):
            img = PILImage.new("RGB", (1, 1), color="red")
            return img
        monkeypatch.setattr(PILImage, "open", fake_open)
        msg = build_multimodal_message("Describe this", b"pngbytes", image_media_type="image/png")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][1]["type"] == "image_url"

    def test_unsupported_media_type_raises(self):
        with pytest.raises(Exception):
            build_multimodal_message("Hi", b"data", image_media_type="text/plain")

    def test_empty_image_bytes_raises(self):
        with pytest.raises(Exception):
            build_multimodal_message("Hi", b"", image_media_type="image/png")
