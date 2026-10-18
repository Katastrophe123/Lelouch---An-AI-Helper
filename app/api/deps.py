"""FastAPI dependency providers."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import UnauthorizedError
from app.config import Settings, get_settings
from app.core.security import AuthError, decode_access_token
from app.db.repository import ConversationRepository, UserRepository
from app.db.session import Database
from app.services.conversation_service import ConversationService
from app.services.tutor_service import TutorService

bearer_scheme = HTTPBearer(auto_error=False, description="JWT from POST /auth/token")


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """The authenticated learner, resolved from the token subject."""

    id: uuid.UUID
    username: str


def get_app_settings(request: Request) -> Settings:
    """Settings stored on app state at startup (falls back to the env)."""
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings or get_settings()


def get_database(request: Request) -> Database:
    database: Database | None = getattr(request.app.state, "database", None)
    if database is None:  # pragma: no cover - would mean a broken lifespan
        raise RuntimeError("Database is not initialised on app state")
    return database


async def get_session(
    database: Annotated[Database, Depends(get_database)],
) -> AsyncIterator[AsyncSession]:
    """Request-scoped session: commits on success, rolls back on error."""
    async with database.session() as session:
        yield session


def get_conversation_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ConversationRepository:
    return ConversationRepository(session)


def get_user_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> UserRepository:
    return UserRepository(session)


def get_conversation_service(
    repository: Annotated[ConversationRepository, Depends(get_conversation_repository)],
) -> ConversationService:
    return ConversationService(repository)


def get_tutor_service(request: Request) -> TutorService:
    service: TutorService | None = getattr(request.app.state, "tutor_service", None)
    if service is None:  # pragma: no cover - would mean a broken lifespan
        raise RuntimeError("TutorService is not initialised on app state")
    return service


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    users: Annotated[UserRepository, Depends(get_user_repository)],
) -> CurrentUser:
    """Validate the bearer token and resolve it to a persisted user.

    The user row is looked up (not trusted from the token) so a token for a
    deleted account stops working immediately.
    """
    if credentials is None or not credentials.credentials:
        raise UnauthorizedError("Missing bearer token")
    try:
        payload = decode_access_token(credentials.credentials, settings)
    except AuthError as exc:
        raise UnauthorizedError(exc.message) from exc

    username = str(payload["sub"])
    user = await users.get_by_username(username)
    if user is None:
        raise UnauthorizedError("Token subject no longer exists")
    return CurrentUser(id=user.id, username=user.username)


class PaginationParams:
    """Shared ``page`` / ``page_size`` query parameters."""

    def __init__(
        self,
        page: Annotated[int, Query(ge=1, description="1-based page number")] = 1,
        page_size: Annotated[int, Query(ge=1, le=100, description="Items per page (max 100)")] = 20,
    ) -> None:
        self.page = page
        self.page_size = page_size


CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
DatabaseDep = Annotated[Database, Depends(get_database)]
UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]
ConversationRepositoryDep = Annotated[ConversationRepository, Depends(get_conversation_repository)]
ConversationServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]
TutorServiceDep = Annotated[TutorService, Depends(get_tutor_service)]
PaginationDep = Annotated[PaginationParams, Depends(PaginationParams)]
