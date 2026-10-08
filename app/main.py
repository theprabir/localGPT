import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.core.logging import (
    create_log_file_path,
    get_logger,
    log_startup,
    setup_logging,
)
from app.core.resource_manager import get_resource_manager
from app.models.database import Database, get_database
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.images import router as images_router
from app.api.system import router as system_router


def create_app(settings=None, logger_instance=None):
    if settings is None:
        settings = get_settings()
    if logger_instance is None:
        log_file = create_log_file_path(settings)
        logger_instance = setup_logging(log_file=log_file)

    log_startup(logger_instance)

    db = get_database(settings=settings)
    rm = get_resource_manager(settings=settings)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0-phase1",
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    app.state.settings = settings
    app.state.db = db
    app.state.rm = rm
    app.state.logger = logger_instance

    # Static files and templates must be served from a known root. Use the package
    # directory as the static root so the UI works without a separate web server.
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    template_dir = os.path.join(os.path.dirname(__file__), "templates")

    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir, check_dir=True), name="static")
    if os.path.isdir(template_dir):
        app.state.templates = Jinja2Templates(directory=template_dir)
        # Version static assets by file mtime so browsers never render a stale
        # UI after an upgrade (paired with the no-cache middleware below).
        app.state.templates.env.globals["asset_version"] = lambda *names: str(
            int(
                max(
                    [
                        os.path.getmtime(os.path.join(static_dir, n))
                        for n in names
                        if os.path.exists(os.path.join(static_dir, n))
                    ]
                    or [0]
                )
            )
        )
    else:
        app.state.templates = None

    app.include_router(chat_router)
    app.include_router(conversations_router)
    app.include_router(images_router)
    app.include_router(system_router)

    @app.middleware("http")
    async def static_asset_revalidation(request: Request, call_next):
        """
        Force revalidation (ETag/304) for static assets. Without this, browsers
        apply heuristic caching (10% of file age) and can keep serving stale
        JS/CSS after an upgrade — which breaks the UI silently.
        """
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        templates = getattr(app.state, "templates", None)
        if templates is None:
            return HTMLResponse("LocalGPT is running", status_code=200)
        return templates.TemplateResponse(request, "chat.html")

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        templates = getattr(app.state, "templates", None)
        if templates is None:
            return HTMLResponse("LocalGPT is running", status_code=200)
        return templates.TemplateResponse(request, "settings.html")

    @app.get("/health")
    async def health():
        from app.api.system import health as system_health

        # NOTE: call with real dependencies. Calling this directly would pass
        # FastAPI Depends placeholders instead of settings/rm objects.
        return system_health(settings=app.state.settings, rm=app.state.rm)

    logger_instance.info("LocalGPT application created")

    return app


def _get_database(request: Request) -> Database:
    return request.app.state.db


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger = getattr(app.state, "logger", None)
    if logger:
        logger.info("LocalGPT shutting down")
    db = getattr(app.state, "db", None)
    if db:
        try:
            db.close()
        except Exception:
            pass


def run():
    settings = get_settings()
    log_file = create_log_file_path(settings)
    logger = setup_logging(log_file=log_file)
    log_startup(logger)

    app = create_app(settings=settings, logger_instance=logger)

    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    run()
