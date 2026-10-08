# UI

LocalGPT's UI is a ChatGPT-style single-page shell with a sidebar and main chat area.

## Features

- Dark and light themes, with system theme detection and persistent selection
- Choice of interface font: default system font, Lato, or EB Garamond — applied to
  the entire UI (headings, chat, history, settings) with per-element weights and
  italics; code blocks stay monospace. Fonts are vendored locally in
  `app/static/fonts/` (no CDN), regenerate with `scripts/download_fonts.py`
- New chat, conversation history grouped by day (Today / Yesterday / date)
- Streaming assistant responses with stop generation and regenerate
- Enter sends the message, Shift+Enter inserts a newline
- Image attachment with preview; uploaded images re-appear in history after reload
- Generated image display and download
- Settings page at `/settings` (appearance, model configuration, status, privacy),
  opened from the gear icon in the sidebar
- Chat panel scrolls internally while the composer stays pinned; the page itself
  does not scroll vertically
- Slim, rounded, theme-aware scrollbars
- Responsive layout with an off-canvas sidebar below 900px (dismiss with the
  backdrop, the toggle, or Escape)

## Frontend stack

HTML, CSS, Alpine.js for small reactive pieces, and htmx for dynamic interactions where used. No build step is required.
