"""Tests for application assembly and the lifespan-managed resources."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.db.session import Database
from app.llm.base import LLMPermanentError
from app.llm.providers import StubLLMClient
from app.main import create_app
from app.services.tutor_service import TutorService


class TestCreateApp:
    def test_routes_are_registered(self, settings: Settings) -> None:
        paths = set(create_app(settings).openapi()["paths"])

        assert {
            "/health",
            "/auth/token",
            "/conversations",
            "/conversations/{conversation_id}",
            "/conversations/{conversation_id}/messages",
            "/conversations/{conversation_id}/messages/stream",
        } <= paths

    def test_settings_are_stored_on_app_state(self, settings: Settings) -> None:
        app = create_app(settings)
        assert app.state.settings is settings


class TestLifespan:
    async def test_lifespan_builds_and_disposes_resources(self, settings: Settings) -> None:
        """Runs the real lifespan against SQLite to verify the wiring."""
        app = create_app(settings)

        async with app.router.lifespan_context(app):
            assert isinstance(app.state.database, Database)
            assert isinstance(app.state.llm, StubLLMClient)
            assert isinstance(app.state.tutor_service, TutorService)
            assert await app.state.database.healthcheck() is True

        assert app.state.database is not None

    async def test_lifespan_fails_fast_without_an_api_key(self, settings: Settings) -> None:
        broken = settings.model_copy(update={"llm_provider": "huggingface", "llm_api_key": None})
        app = create_app(broken)

        with pytest.raises(LLMPermanentError, match="LLM_API_KEY"):
            async with app.router.lifespan_context(app):
                pass  # pragma: no cover - the context must not open

    async def test_health_works_through_the_real_lifespan(self, settings: Settings) -> None:
        app = create_app(settings)

        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as http,
        ):
            response = await http.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["llm_provider"] == "stub"
        assert body["llm_model"] == "stub-tutor-v1"
