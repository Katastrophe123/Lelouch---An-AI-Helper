"""Unauthenticated liveness/readiness endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from app import __version__
from app.api.deps import SettingsDep, get_database
from app.db.session import Database
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service and dependency health",
    responses={503: {"description": "A dependency is unavailable"}},
)
async def health(
    request: Request,
    response: Response,
    settings: SettingsDep,
    database: Annotated[Database, Depends(get_database)],
) -> HealthResponse:
    """Report process health plus database reachability.

    Returns 503 when the database is unreachable so orchestrators can act on
    it, while still returning a parseable body for debugging.
    """
    db_ok = await database.healthcheck()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    llm = getattr(request.app.state, "llm", None)
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        app=settings.app_name,
        environment=settings.environment,
        version=__version__,
        database="up" if db_ok else "down",
        llm_provider=settings.llm_provider,
        llm_model=getattr(llm, "model", settings.llm_model),
    )
