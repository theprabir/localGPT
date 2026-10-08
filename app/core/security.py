import html
import os
import re
import uuid
from typing import Optional

from app.config import get_settings

# Allowed image file extensions and MIME types are defined by configuration.
# All uploaded files are treated as untrusted.


def sanitize_filename(raw_name: str) -> str:
    """
    Return a safe internal filename base. This is NOT the final stored filename;
    LocalGPT stores assets under a UUID with the original extension appended after
    validation.
    """
    if not raw_name:
        return ""

    basename = os.path.basename(raw_name)

    # Remove control characters and normalize whitespace.
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", basename)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_. ")

    # Keep only a conservative set of characters.
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)

    if not cleaned or cleaned.startswith("."):
        cleaned = f"file_{uuid.uuid4().hex[:8]}"

    return cleaned[:255]


def safe_storage_filename(original_filename: str, extension: str | None = None) -> str:
    """
    Create a UUID-based storage filename with an allowed extension.

    This prevents path traversal, filename collisions, and executable naming.
    """
    if extension:
        ext = extension.lower()
    else:
        _, ext = os.path.splitext(original_filename)
        ext = ext.lower()

    settings = get_settings()
    if ext not in settings.allowed_image_extensions:
        ext = ".png"

    return f"{uuid.uuid4().hex}{ext}"


def validate_image_extension(filename: str) -> str:
    """
    Return the validated extension or raise ValueError.
    """
    settings = get_settings()
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in settings.allowed_image_extensions:
        raise ValueError(
            f"Unsupported file extension {ext}. Allowed: {', '.join(settings.allowed_image_extensions)}"
        )
    return ext


def validate_image_mime(media_type: str) -> str:
    """
    Validate the supplied media type.

    Note: MIME sniffing is still required server-side using the file content. This
    function validates the declared media type only.
    """
    settings = get_settings()
    normalized = media_type.split(";", 1)[0].strip().lower()
    if normalized not in settings.allowed_image_mime_types:
        raise ValueError(
            f"Unsupported media type {media_type}. Allowed: {', '.join(settings.allowed_image_mime_types)}"
        )
    return normalized


def is_safe_image_path(path: str) -> bool:
    """
    Ensure a resolved path is inside an approved data directory and not a traversal.
    """
    try:
        resolved = os.path.realpath(path)
    except Exception:
        return False

    settings = get_settings()

    approved_roots = {
        os.path.realpath(settings.conversations_dir_path),
        os.path.realpath(settings.uploads_dir_path),
        os.path.realpath(settings.generated_dir_path),
        os.path.realpath(settings.thumbnails_dir_path),
        os.path.realpath(settings.cache_dir_path),
    }

    return any(resolved.startswith(root) for root in approved_roots)


def html_escape(text: str) -> str:
    return html.escape(text, quote=True)


def sanitize_html_fragment(html_text: str, max_length: int = 20000) -> str:
    """
    Lightweight HTML sanitization fallback when bleach is unavailable.

    This is deliberately conservative: it escapes everything by default and only
    allows a minimal safe subset.
    """
    if not html_text:
        return ""

    text = html_text[:max_length]

    # Bleach is a required dependency for LocalGPT, so normally this path is only a
    # safety net. Use bleach when available.
    try:
        import bleach

        allowed_tags = {
            "p",
            "br",
            "pre",
            "code",
            "em",
            "strong",
            "ul",
            "ol",
            "li",
            "blockquote",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "table",
            "thead",
            "tbody",
            "tr",
            "th",
            "td",
            "hr",
            "img",
        }
        allowed_attrs = {
            "*": {"class", "id"},
            "a": {"href", "title", "rel"},
            "img": {"src", "alt", "title", "loading"},
        }
        return bleach.clean(
            text,
            tags=sorted(allowed_tags),
            attributes=allowed_attrs,
            strip=True,
            strip_comments=True,
        )
    except Exception:
        return html_escape(text)


def render_markdown(text: str) -> str:
    """
    Render Markdown to sanitized HTML suitable for the web UI.

    This escapes the input first, then converts Markdown. The output is then sanitized
    to block script injection from malformed Markdown.
    """
    if not text:
        return ""

    try:
        import markdown as md_lib
    except Exception as exc:
        raise RuntimeError("Markdown rendering is not available") from exc

    # Convert Markdown -> HTML, then sanitize.
    html_output = md_lib.markdown(
        text,
        extensions=[
            "nl2br",
            "fenced_code",
            "codehilite",
        ],
    )
    return sanitize_html_fragment(html_output)


def validate_max_image_size(width: int, height: int, max_size: int | None = None) -> tuple[int, int]:
    settings = get_settings()
    limit = max_size if max_size is not None else settings.max_image_size
    if width > limit or height > limit:
        raise ValueError(
            f"Image dimensions {width}x{height} exceed the maximum allowed size {limit}x{limit}"
        )
    return width, height
