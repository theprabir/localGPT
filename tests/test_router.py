import pytest

from app.core.router import classify_intent, classify_intent_structured, extract_image_prompt_from_request, extract_image_edit_params, detect_inline_image_token_provider


class TestClassifyIntent:
    def test_plain_chat(self):
        intent = classify_intent("Hello, how are you?")
        assert intent.intent == "CHAT"
        assert intent.prompt == "Hello, how are you?"

    def test_create_image_request(self):
        intent = classify_intent("Create an image of a dragon")
        assert intent.intent == "IMAGE_GENERATION"
        assert intent.prompt == "Create an image of a dragon"

    def test_image_analysis_with_attachment(self):
        intent = classify_intent("What is this?", has_attachment=True)
        assert intent.intent == "IMAGE_ANALYSIS"

    def test_image_edit_with_attachment(self):
        intent = classify_intent("Make the car red", has_attachment=True)
        assert intent.intent == "IMAGE_EDIT"
        assert intent.operation == "img2img"

    def test_generate_keyword(self):
        intent = classify_intent("Generate a picture of mountains")
        assert intent.intent == "IMAGE_GENERATION"

    def test_edit_keyword_no_attachment(self):
        intent = classify_intent("Change the color of the car")
        assert intent.intent == "IMAGE_EDIT"

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            classify_intent("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            classify_intent("   ")


class TestClassifyIntentStructured:
    def test_valid_structured_chat(self):
        intent = classify_intent_structured({
            "intent": "CHAT",
            "prompt": "Explain this",
            "raw_text": "Explain this",
        })
        assert intent.intent == "CHAT"

    def test_valid_structured_image_generation(self):
        intent = classify_intent_structured({
            "intent": "IMAGE_GENERATION",
            "prompt": "A sunset",
            "width": 512,
            "height": 512,
            "raw_text": "A sunset",
        })
        assert intent.intent == "IMAGE_GENERATION"
        assert intent.width == 512

    def test_valid_structured_image_edit(self):
        intent = classify_intent_structured({
            "intent": "IMAGE_EDIT",
            "prompt": "Make it red",
            "operation": "img2img",
            "strength": 0.6,
            "raw_text": "Make it red",
        })
        assert intent.intent == "IMAGE_EDIT"
        assert intent.operation == "img2img"
        assert intent.strength == 0.6

    def test_invalid_intent_rejected(self):
        with pytest.raises(ValueError):
            classify_intent_structured({"intent": "BOGUS", "raw_text": "x"})

    def test_missing_prompt_rejected(self):
        with pytest.raises(ValueError):
            classify_intent_structured({"intent": "IMAGE_GENERATION", "raw_text": "x"})


class TestExtractImagePromptFromRequest:
    def test_normal_prompt(self):
        assert extract_image_prompt_from_request("Draw a tree") == "Draw a tree"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            extract_image_prompt_from_request("")


class TestExtractImageEditParams:
    def test_default_strength(self):
        params = extract_image_edit_params("Make the sky purple")
        assert params["operation"] == "img2img"
        assert params["strength"] == pytest.approx(0.55)

    def test_custom_strength(self):
        params = extract_image_edit_params("Make the sky purple strength=0.8")
        assert params["strength"] == pytest.approx(0.8)


class TestDetectInlineImageTokenProvider:
    def test_present(self):
        assert detect_inline_image_token_provider("create an image") is True

    def test_absent(self):
        assert detect_inline_image_token_provider("hello world") is False
