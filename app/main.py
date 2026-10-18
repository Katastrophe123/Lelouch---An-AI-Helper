"""FastAPI application factory and lifespan wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import auth, conversations, health
from app.config import Settings, get_settings
from app.db.session import Database
from app.llm.base import LLMClient
from app.llm.factory import build_llm_client
from app.logging_config import configure_logging, get_logger
from app.services.tutor_service import TutorService

logger = get_logger(__name__)

DESCRIPTION = """
A LangGraph tutoring agent behind a stateless HTTP API.

Each question runs **understand -> solve -> self-check -> explain**. If the
self-check rejects the answer the graph retries solving (bounded), and once
retries are spent it returns an explicitly *unverified* partial answer rather
than passing a rejected answer off as checked. Users, conversations and
messages live in PostgreSQL, so history survives restarts and is queryable.

All endpoints except `/health` and `/auth/token` require a bearer token from
`POST /auth/token`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build process-wide resources once and tear them down on shutdown."""
    settings: Settings = app.state.settings
    database = Database.from_settings(settings)
    llm: LLMClient = build_llm_client(settings)

    app.state.database = database
    app.state.llm = llm
    app.state.tutor_service = TutorService(settings=settings, llm=llm)

    logger.info(
        "application_started",
        app=settings.app_name,
        version=__version__,
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        llm_model=llm.model,
        streaming=llm.supports_streaming,
    )
    try:
        yield
    finally:
        await llm.aclose()
        await database.dispose()
        logger.info("application_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app. Tests call this with overridden settings."""
    resolved = settings or get_settings()
    configure_logging(level=resolved.log_level, json_output=resolved.log_json)

    app = FastAPI(
        title="AI Homework Helper",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = resolved

    app.add_middleware(RequestContextMiddleware)
    # The Streamlit frontend is a separate origin; it calls the API server-side,
    # but CORS is enabled so a browser-based client works too.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not resolved.is_production else [resolved.api_base_url],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(conversations.router)

    return app


app = create_app()
