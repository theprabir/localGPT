# Development

## Local runs

Use `run_localgpt.py` during development. Set `.env` values as needed.

## Testing

Run `pytest` from the project root. Tests use mocks for inference and do not require the full models to be loaded.

## Frontend

Frontend assets are served from `app/static`. htmx (`app/static/vendor/htmx.js`, v1.9.12) and Alpine.js (`app/static/vendor/alpine.js`, v3.14.9) are fully vendored locally, so the UI works offline after installation. `app.js` must stay ordered before `alpine.js` in `base.html` so component registrations exist when Alpine initializes.
