"""
CareerCopilot V2 — FastAPI application entrypoint.

Start with:
  uvicorn src.main:app --reload        (development)
  gunicorn src.main:app -k uvicorn.workers.UvicornWorker  (production)
"""

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.health import router as health_router
from src.api.jd import router as jd_router
from src.api.middleware import RequestLoggingMiddleware
from src.api.pages import router as pages_router
from src.api.resume import router as resume_router
from src.api.roles import router as roles_router
from src.api.tasks import router as tasks_router
from src.config import get_settings
from src.database import Base, engine

log = structlog.get_logger(__name__)
settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(
        title="CareerCopilot",
        description="AI-powered skill gap analysis and career roadmap generator.",
        version="2.0.0",
        docs_url="/api/docs" if settings.debug else None,
        redoc_url=None,
    )

    # --- Middleware ---
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Static files ---
    app.mount("/static", StaticFiles(directory="src/static"), name="static")

    # --- Routers ---
    app.include_router(health_router)
    app.include_router(pages_router)
    app.include_router(resume_router)
    app.include_router(jd_router)
    app.include_router(roles_router)
    app.include_router(tasks_router)

    # --- Startup ---
    @app.on_event("startup")
    async def on_startup():
        log.info("app_startup", env=settings.environment, model=settings.llm_model)
        # Create tables if they don't exist (migrations handle schema in prod)
        if settings.debug:
            Base.metadata.create_all(bind=engine)

    # --- 404 handler ---
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        from fastapi.templating import Jinja2Templates
        templates = Jinja2Templates(directory="src/templates")
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": "Page not found."},
            status_code=404,
        )

    # --- 500 handler ---
    @app.exception_handler(500)
    async def server_error_handler(request: Request, exc):
        log.exception("unhandled_server_error", path=request.url.path)
        from fastapi.templating import Jinja2Templates
        templates = Jinja2Templates(directory="src/templates")
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": "Something went wrong on our end. Please try again."},
            status_code=500,
        )

    return app


app = create_app()
