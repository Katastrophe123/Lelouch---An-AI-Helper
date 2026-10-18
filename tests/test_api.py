"""HTTP-level tests: auth, ownership isolation, status codes, SSE, pagination."""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.errors import REQUEST_ID_HEADER
from app.config import Settings
from app.db.session import Database
from app.models.enums import AnswerStatus
from app.services.tutor_service import TutorService
from tests.conftest import TEST_PASSWORD
from tests.fakes import (
    EXPLANATION,
    ScriptedLLMClient,
    StreamingScriptedLLMClient,
    build_self_check,
    build_solution,
    build_understanding,
    happy_script,
)

QUESTION = "A ball is dropped from 20 m. How long until it hits the ground?"


def rewire(app: FastAPI, settings: Settings, llm: Any) -> None:
    """Swap the app's LLM and service, for tests that need a different script."""
    app.state.llm = llm
    app.state.tutor_service = TutorService(settings=settings, llm=llm)


class TestHealth:
    async def test_health_needs_no_token(self, client: AsyncClient) -> None:
        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["database"] == "up"
        assert body["environment"] == "test"
        assert body["llm_model"] == "scripted-model"

    async def test_health_reports_degraded_when_the_database_is_down(
        self, app: FastAPI, database: Database
    ) -> None:
        class DeadDatabase:
            async def healthcheck(self) -> bool:
                return False

        app.state.database = DeadDatabase()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as http:
            response = await http.get("/health")

        assert response.status_code == 503
        assert response.json()["database"] == "down"

    async def test_response_carries_a_request_id(self, client: AsyncClient) -> None:
        assert (await client.get("/health")).headers[REQUEST_ID_HEADER]

    async def test_inbound_request_id_is_echoed(self, client: AsyncClient) -> None:
        response = await client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc"})
        assert response.headers[REQUEST_ID_HEADER] == "trace-abc"


class TestAuth:
    async def test_first_sign_in_creates_the_learner(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/token", json={"username": "newbie", "password": TEST_PASSWORD}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "newbie"
        assert uuid.UUID(body["user_id"])
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == 1800

    async def test_second_sign_in_reuses_the_same_user(self, client: AsyncClient) -> None:
        first = await client.post(
            "/auth/token", json={"username": "ada", "password": TEST_PASSWORD}
        )
        second = await client.post(
            "/auth/token", json={"username": "ada", "password": TEST_PASSWORD}
        )
        assert first.json()["user_id"] == second.json()["user_id"]

    async def test_wrong_password_returns_401(self, client: AsyncClient) -> None:
        response = await client.post("/auth/token", json={"username": "ada", "password": "nope"})

        assert response.status_code == 401
        body = response.json()
        assert body["error"]["code"] == "unauthorized"
        assert body["request_id"]
        assert response.headers["WWW-Authenticate"] == "Bearer"

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"username": "ada"},
            {"password": TEST_PASSWORD},
            {"username": "", "password": TEST_PASSWORD},
            {"username": "bad name!", "password": TEST_PASSWORD},
        ],
    )
    async def test_invalid_payloads_return_422(
        self, client: AsyncClient, payload: dict[str, str]
    ) -> None:
        response = await client.post("/auth/token", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"


class TestAuthEnforcement:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/conversations"),
            ("post", "/conversations"),
            ("get", f"/conversations/{uuid.uuid4()}"),
            ("post", f"/conversations/{uuid.uuid4()}/messages"),
        ],
    )
    async def test_protected_routes_reject_missing_tokens(
        self, client: AsyncClient, method: str, path: str
    ) -> None:
        kwargs = {"json": {}} if method == "post" else {}
        response = await getattr(client, method)(path, **kwargs)

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    async def test_rejects_a_malformed_token(self, client: AsyncClient) -> None:
        response = await client.get("/conversations", headers={"Authorization": "Bearer junk"})
        assert response.status_code == 401

    async def test_rejects_a_non_bearer_scheme(self, client: AsyncClient) -> None:
        response = await client.get(
            "/conversations", headers={"Authorization": "Basic dXNlcjpwYXNz"}
        )
        assert response.status_code == 401

    async def test_token_for_an_unknown_user_is_rejected(
        self, client: AsyncClient, token_for: Any
    ) -> None:
        """A validly-signed token whose subject has no row must not work."""
        token = token_for("ghost")
        response = await client.get("/conversations", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 401
        assert "no longer exists" in response.json()["error"]["message"]


class TestCreateConversation:
    async def test_creates_an_empty_conversation(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            "/conversations", json={"title": "Calculus", "subject": "math"}, headers=auth_headers
        )

        assert response.status_code == 201
        body = response.json()
        assert body["title"] == "Calculus"
        assert body["subject"] == "math"
        assert body["status"] == "active"
        assert body["message_count"] == 0
        assert body["messages"] == []
        assert uuid.UUID(body["id"])

    async def test_defaults_when_nothing_is_supplied(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        body = (await client.post("/conversations", json={}, headers=auth_headers)).json()
        assert body["title"] == "New conversation"
        assert body["subject"] is None

    async def test_invalid_subject_returns_422(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            "/conversations", json={"subject": "astrology"}, headers=auth_headers
        )
        assert response.status_code == 422


class TestPostMessage:
    async def test_returns_both_messages_and_the_trace(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
    ) -> None:
        response = await client.post(
            f"/conversations/{api_conversation['id']}/messages",
            json={"content": QUESTION},
            headers=auth_headers,
        )

        assert response.status_code == 201
        body = response.json()
        assert body["user_message"]["role"] == "user"
        assert body["user_message"]["content"] == QUESTION
        assistant = body["assistant_message"]
        assert assistant["role"] == "assistant"
        assert assistant["answer_status"] == AnswerStatus.VERIFIED.value
        assert assistant["content"] == EXPLANATION
        assert [step["node"] for step in assistant["trace"]] == [
            "understand",
            "solve",
            "self_check",
            "explain",
        ]

    async def test_history_persists_across_requests(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
    ) -> None:
        """The v1 -> v2 headline: a reload shows the same conversation."""
        await client.post(
            f"/conversations/{api_conversation['id']}/messages",
            json={"content": QUESTION},
            headers=auth_headers,
        )

        reloaded = await client.get(
            f"/conversations/{api_conversation['id']}", headers=auth_headers
        )

        body = reloaded.json()
        assert body["message_count"] == 2
        assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
        assert body["subject"] == "math"  # backfilled from the answer

    async def test_unverified_answer_is_labelled_as_such(
        self,
        app: FastAPI,
        settings: Settings,
        client: AsyncClient,
        auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
    ) -> None:
        rewire(
            app,
            settings,
            ScriptedLLMClient(
                {
                    "understand": [build_understanding()],
                    "solve": [build_solution(answer="a long but rejected answer here")],
                    "self_check": [build_self_check(passed=False, issues=["Wrong formula."])],
                    "explain": [EXPLANATION],
                }
            ),
        )

        response = await client.post(
            f"/conversations/{api_conversation['id']}/messages",
            json={"content": QUESTION},
            headers=auth_headers,
        )

        assistant = response.json()["assistant_message"]
        assert assistant["answer_status"] == AnswerStatus.UNVERIFIED.value
        assert assistant["fallback_reason"] == "self_check_failed"
        assert assistant["retry_count"] == 2
        assert "could not fully verify" in assistant["content"]

    @pytest.mark.parametrize("payload", [{}, {"content": ""}, {"content": "hi"}])
    async def test_invalid_payloads_return_422(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
        payload: dict[str, str],
    ) -> None:
        response = await client.post(
            f"/conversations/{api_conversation['id']}/messages",
            json=payload,
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_oversized_question_is_rejected(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
    ) -> None:
        response = await client.post(
            f"/conversations/{api_conversation['id']}/messages",
            json={"content": "x" * 8001},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_unknown_conversation_returns_404(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        missing = uuid.uuid4()
        response = await client.post(
            f"/conversations/{missing}/messages",
            json={"content": QUESTION},
            headers=auth_headers,
        )

        assert response.status_code == 404
        assert response.json()["error"]["details"]["conversation_id"] == str(missing)


class TestOwnershipIsolation:
    async def test_another_learner_cannot_read_a_conversation(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        other_auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
    ) -> None:
        response = await client.get(
            f"/conversations/{api_conversation['id']}", headers=other_auth_headers
        )
        assert response.status_code == 404

    async def test_another_learner_cannot_post_to_a_conversation(
        self,
        client: AsyncClient,
        other_auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
    ) -> None:
        response = await client.post(
            f"/conversations/{api_conversation['id']}/messages",
            json={"content": QUESTION},
            headers=other_auth_headers,
        )
        assert response.status_code == 404

    async def test_another_learner_cannot_stream_to_a_conversation(
        self,
        client: AsyncClient,
        other_auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
    ) -> None:
        response = await client.post(
            f"/conversations/{api_conversation['id']}/messages/stream",
            json={"content": QUESTION},
            headers=other_auth_headers,
        )
        assert response.status_code == 404

    async def test_listings_are_scoped_per_learner(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        other_auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
    ) -> None:
        mine = (await client.get("/conversations", headers=auth_headers)).json()
        theirs = (await client.get("/conversations", headers=other_auth_headers)).json()

        assert mine["total"] == 1
        assert theirs["total"] == 0


class TestGetConversation:
    async def test_unknown_id_returns_a_404_envelope(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        missing = uuid.uuid4()
        response = await client.get(f"/conversations/{missing}", headers=auth_headers)

        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"] == "not_found"
        assert body["error"]["details"]["conversation_id"] == str(missing)

    async def test_non_uuid_id_returns_422(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.get("/conversations/not-a-uuid", headers=auth_headers)
        assert response.status_code == 422


class TestListConversations:
    async def _seed(self, client: AsyncClient, headers: dict[str, str], count: int) -> None:
        for index in range(count):
            response = await client.post(
                "/conversations", json={"title": f"Thread {index}"}, headers=headers
            )
            assert response.status_code == 201

    async def test_empty_listing(self, client: AsyncClient, auth_headers: dict[str, str]) -> None:
        body = (await client.get("/conversations", headers=auth_headers)).json()
        assert body == {"items": [], "page": 1, "page_size": 20, "total": 0, "total_pages": 0}

    async def test_pagination_metadata_is_correct(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        await self._seed(client, auth_headers, 5)

        body = (await client.get("/conversations?page=2&page_size=2", headers=auth_headers)).json()

        assert body["page"] == 2
        assert body["page_size"] == 2
        assert body["total"] == 5
        assert body["total_pages"] == 3
        assert len(body["items"]) == 2

    async def test_summaries_omit_the_message_list(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        await self._seed(client, auth_headers, 1)
        body = (await client.get("/conversations", headers=auth_headers)).json()
        assert "messages" not in body["items"][0]

    async def test_filters_by_subject(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        await client.post("/conversations", json={"subject": "math"}, headers=auth_headers)
        await client.post("/conversations", json={"subject": "physics"}, headers=auth_headers)

        body = (await client.get("/conversations?subject=math", headers=auth_headers)).json()
        assert body["total"] == 1

    @pytest.mark.parametrize(
        "query", ["page=0", "page_size=0", "page_size=500", "status=nonsense", "subject=nope"]
    )
    async def test_invalid_query_parameters_return_422(
        self, client: AsyncClient, auth_headers: dict[str, str], query: str
    ) -> None:
        response = await client.get(f"/conversations?{query}", headers=auth_headers)
        assert response.status_code == 422


class TestStreamingEndpoint:
    @staticmethod
    def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
        """Decode a raw ``text/event-stream`` body into (event, data) pairs."""
        events: list[tuple[str, dict[str, Any]]] = []
        name: str | None = None
        for line in text.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and name:
                payload = line.split(":", 1)[1].strip()
                with contextlib.suppress(json.JSONDecodeError):
                    events.append((name, json.loads(payload)))
        return events

    async def test_streams_steps_tokens_and_done(
        self,
        app: FastAPI,
        settings: Settings,
        client: AsyncClient,
        auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
    ) -> None:
        rewire(app, settings, StreamingScriptedLLMClient(happy_script()))

        async with client.stream(
            "POST",
            f"/conversations/{api_conversation['id']}/messages/stream",
            json={"content": QUESTION},
            headers=auth_headers,
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = await response.aread()

        events = self._parse_sse(body.decode())
        names = [name for name, _ in events]

        assert names.count("step") == 4
        assert names.count("token") > 1
        assert names.count("done") == 1
        assert names.index("step") < names.index("token") < names.index("done")

        done = events[-1][1]
        assert done["answer_status"] == AnswerStatus.VERIFIED.value
        assert done["content"] == EXPLANATION
        assert uuid.UUID(done["assistant_message_id"])

    async def test_streamed_turn_is_persisted(
        self,
        app: FastAPI,
        settings: Settings,
        client: AsyncClient,
        auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
    ) -> None:
        rewire(app, settings, StreamingScriptedLLMClient(happy_script()))

        async with client.stream(
            "POST",
            f"/conversations/{api_conversation['id']}/messages/stream",
            json={"content": QUESTION},
            headers=auth_headers,
        ) as response:
            await response.aread()

        reloaded = (
            await client.get(f"/conversations/{api_conversation['id']}", headers=auth_headers)
        ).json()
        assert reloaded["message_count"] == 2
        assert reloaded["messages"][1]["answer_status"] == AnswerStatus.VERIFIED.value

    async def test_token_chunks_reassemble_to_the_stored_answer(
        self,
        app: FastAPI,
        settings: Settings,
        client: AsyncClient,
        auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
    ) -> None:
        rewire(app, settings, StreamingScriptedLLMClient(happy_script()))

        async with client.stream(
            "POST",
            f"/conversations/{api_conversation['id']}/messages/stream",
            json={"content": QUESTION},
            headers=auth_headers,
        ) as response:
            body = await response.aread()

        events = self._parse_sse(body.decode())
        tokens = "".join(data["text"] for name, data in events if name == "token")
        done = next(data for name, data in events if name == "done")
        assert tokens.strip() == done["content"]

    async def test_streaming_requires_a_token(
        self, client: AsyncClient, api_conversation: dict[str, Any]
    ) -> None:
        response = await client.post(
            f"/conversations/{api_conversation['id']}/messages/stream",
            json={"content": QUESTION},
        )
        assert response.status_code == 401

    async def test_unknown_conversation_is_a_json_404_not_a_stream(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Ownership is checked before the stream opens."""
        response = await client.post(
            f"/conversations/{uuid.uuid4()}/messages/stream",
            json={"content": QUESTION},
            headers=auth_headers,
        )

        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"]["code"] == "not_found"


class TestOpenAPI:
    async def test_schema_is_served(self, client: AsyncClient) -> None:
        schema = (await client.get("/openapi.json")).json()

        assert "/conversations" in schema["paths"]
        assert "/conversations/{conversation_id}" in schema["paths"]
        assert "/conversations/{conversation_id}/messages" in schema["paths"]
        assert "/conversations/{conversation_id}/messages/stream" in schema["paths"]
        assert "/health" in schema["paths"]

    async def test_protected_routes_declare_security(self, client: AsyncClient) -> None:
        schema = (await client.get("/openapi.json")).json()
        assert "security" in schema["paths"]["/conversations"]["post"]
        assert "security" not in schema["paths"]["/health"]["get"]
