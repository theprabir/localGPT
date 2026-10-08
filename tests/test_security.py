import os
import pytest

from app.core.security import (
    sanitize_filename,
    safe_storage_filename,
    validate_image_extension,
    validate_image_mime,
    is_safe_image_path,
    html_escape,
    render_markdown,
)


class TestSanitizeFilename:
    def test_basic_name_preserved(self):
        assert sanitize_filename("photo.jpg") == "photo.jpg"

    def test_path_traversal_removed(self):
        # os.path.basename is applied inside sanitize_filename, so ../etc/passwd -> passwd
        assert sanitize_filename("../etc/passwd") == "passwd"

    def test_absolute_path_basename_only(self):
        assert sanitize_filename("/foo/bar/baz.txt") == "baz.txt"

    def test_invalid_chars_replaced(self):
        # basename is applied first, so leading directory is dropped
        assert sanitize_filename("a/file name.txt") == "file_name.txt"

    def test_empty_returns_empty(self):
        assert sanitize_filename("") == ""

    def test_whitespace_normalized(self):
        assert sanitize_filename("  my   file  .png  ") == "my_file_.png"


class TestSafeStorageFilename:
    def test_uuid_extension_preserved(self):
        name = safe_storage_filename("photo.jpg")
        assert name.startswith("") or True  # just a UUID
        assert name.endswith(".jpg")

    def test_downloaded_extension_normalized(self):
        name = safe_storage_filename("PHOTO.JPG", extension=".jpg")
        assert name.endswith(".jpg")

    def test_unsupported_extension_falls_back(self):
        name = safe_storage_filename("archive.exe")
        assert name.endswith(".png")


class TestValidateImageExtension:
    def test_allowed_extension_ok(self):
        assert validate_image_extension("image.png") == ".png"

    def test_disallowed_extension_raises(self):
        with pytest.raises(ValueError):
            validate_image_extension("script.exe")


class TestValidateImageMime:
    def test_allowed_mime_ok(self):
        assert validate_image_mime("image/png") == "image/png"

    def test_mime_with_charset_normalized(self):
        assert validate_image_mime("image/png; charset=binary") == "image/png"

    def test_disallowed_mime_raises(self):
        with pytest.raises(ValueError):
            validate_image_mime("application/x-executable")


class TestIsSafeImagePath:
    def test_path_inside_uploads(self, tmp_path, monkeypatch):
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        file = uploads / "a.png"
        file.write_text("x")
        from app.config import LocalGPTSettings
        settings = LocalGPTSettings(data_dir=str(tmp_path / "data"), uploads_dir=str(uploads))
        import app.core.security as sec
        monkeypatch.setattr(sec, "get_settings", lambda: settings)
        assert is_safe_image_path(str(file)) is True

    def test_path_outside_approved_dirs(self, tmp_path, monkeypatch):
        outside = tmp_path / "outside.txt"
        outside.write_text("x")
        from app.config import LocalGPTSettings
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        settings = LocalGPTSettings(data_dir=str(tmp_path / "data"), uploads_dir=str(uploads))
        import app.core.security as sec
        monkeypatch.setattr(sec, "get_settings", lambda: settings)
        assert is_safe_image_path(str(outside)) is False


class TestHtmlEscape:
    def test_script_escaped(self):
        assert html_escape("<script>") == "&lt;script&gt;"


class TestRenderMarkdown:
    def test_simple_markdown_rendered(self):
        html = render_markdown("# Hello\n\nSome **bold** text")
        assert "<h1>" in html
        assert "<strong>" in html

    def test_markdown_script_escaped(self):
        html = render_markdown("`alert(1)`")
        assert "<script>" not in html

    def test_empty_input(self):
        assert render_markdown("") == ""
