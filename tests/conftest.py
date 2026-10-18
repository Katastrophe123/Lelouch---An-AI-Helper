"""Shared fixtures.

The whole suite runs offline: SQLite in-memory for persistence and scripted
fake LLM clients for the model. No API key, no PostgreSQL, no network.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.agents.graph import build_tutor_graph
from app.config import Settings
from app.core.security import create_access_token
from app.db.base import Base
from app.db.models import Conversation, User
from app.db.repository import ConversationRepository, UserRepository
from app.db.session import Database
from app.llm.base import LLMClient
from app.main import create_app
from app.services.conversation_service import ConversationService
from app.services.tutor_service import TutorService
from tests.fakes import ScriptedLLMClient, happy_script

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_PASSWORD = "test-password"


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings: no backoff delay, stub provider, fixed secret."""
    return Settings(
        environment="test",
        log_level="WARNING",
        log_json=False,
        database_url=TEST_DATABASE_URL,
        jwt_secret="test-secret-value-for-unit-tests-only",
        jwt_expire_minutes=30,
        auth_password=TEST_PASSWORD,
        llm_provider="stub",
        llm_max_attempts=3,
        llm_backoff_base_seconds=0.0,
        self_check_min_confidence=0.6,
        max_solve_retries=2,
        max_graph_steps=14,
        history_window=8,
    )


@pytest.fixture
async def database(settings: Settings) -> AsyncIterator[Database]:
    """In-memory database with the schema created from the ORM metadata."""
    db = Database(TEST_DATABASE_URL, echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield db
    finally:
        await db.dispose()


@pytest.fixture
async def session(database: Database) -> AsyncIterator[Any]:
    """A committing session scope for repository-level tests."""
    async with database.session() as sess:
        yield sess


@pytest.fixture
async def users(session: Any) -> UserRepository:
    return UserRepository(session)


@pytest.fixture
async def repository(session: Any) -> ConversationRepository:
    return ConversationRepository(session)


@pytest.fixture
async def conversation_service(repository: ConversationRepository) -> ConversationService:
    return ConversationService(repository)


@pytest.fixture
async def user(users: UserRepository) -> User:
    return await users.get_or_create("ada", display_name="Ada")


@pytest.fixture
async def other_user(users: UserRepository) -> User:
    return await users.get_or_create("grace", display_name="Grace")


@pytest.fixture
async def conversation(repository: ConversationRepository, user: User) -> Conversation:
    return await repository.create_conversation(user_id=user.id, title="Physics revision")


@pytest.fixture
def sleep_calls() -> list[float]:
    """Collects backoff delays so retry timing is assertable without waiting."""
    return []


@pytest.fixture
def fake_sleep(sleep_calls: list[float]) -> Any:
    async def _sleep(delay: float) -> None:
        sleep_calls.append(delay)

    return _sleep


@pytest.fixture
def happy_llm() -> ScriptedLLMClient:
    """Understands, solves, passes the self-check and explains."""
    return ScriptedLLMClient(happy_script())


@pytest.fixture
def graph_factory(settings: Settings, fake_sleep: Any) -> Any:
    """Builds a compiled graph around a supplied LLM client."""

    def _factory(
        llm: LLMClient,
        *,
        overrides: dict[str, Any] | None = None,
        emitter: Any = None,
    ) -> Any:
        resolved = settings.model_copy(update=overrides or {})
        return build_tutor_graph(settings=resolved, llm=llm, sleep=fake_sleep, emitter=emitter)

    return _factory


@pytest.fixture
def tutor_service(
    settings: Settings, happy_llm: ScriptedLLMClient, fake_sleep: Any
) -> TutorService:
    return TutorService(settings=settings, llm=happy_llm, sleep=fake_sleep)


@pytest.fixture
def app(
    settings: Settings,
    database: Database,
    happy_llm: ScriptedLLMClient,
    fake_sleep: Any,
) -> Iterator[FastAPI]:
    """App instance wired to the test database and a scripted LLM.

    ``create_app`` is used so routing, middleware and error handlers are the
    real ones; only the lifespan-managed resources are substituted.
    """
    application = create_app(settings)
    application.state.database = database
    application.state.llm = happy_llm
    application.state.tutor_service = TutorService(
        settings=settings, llm=happy_llm, sleep=fake_sleep
    )
    yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        yield http_client


@pytest.fixture
def token_for(settings: Settings) -> Any:
    """Mint a token for an arbitrary username without hitting the endpoint."""

    def _token(username: str, user_id: uuid.UUID | None = None) -> str:
        claims = {"uid": str(user_id)} if user_id else None
        token, _ = create_access_token(username, settings, extra_claims=claims)
        return token

    return _token


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    """Sign in through the real endpoint so the user row exists."""
    response = await client.post("/auth/token", json={"username": "ada", "password": TEST_PASSWORD})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
async def other_auth_headers(client: AsyncClient) -> dict[str, str]:
    """A second learner, for ownership-isolation tests."""
    response = await client.post(
        "/auth/token", json={"username": "grace", "password": TEST_PASSWORD}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
async def api_conversation(client: AsyncClient, auth_headers: dict[str, str]) -> dict[str, Any]:
    response = await client.post(
        "/conversations", json={"title": "Physics revision"}, headers=auth_headers
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


QUESTION = "A ball is dropped from 20 m. How long until it hits the ground?"
