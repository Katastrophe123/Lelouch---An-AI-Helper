"""Agent-mode conversations over HTTP, using a scripted agent runner."""

from __future__ import annotations

import json
from typing import Any

import pytest
from httpx import AsyncClient

from app.assistant.events import TurnRecorder
from app.assistant.runner import AgentOutcome, AgentRequest, AgentRunner, DocumentSearch
from app.models.enums import NodeName

QUESTION = "When was the Battle of Plassey?"


class RecordingRunner(AgentRunner):
    """Scripted runner that records requests and searches the real knowledge base."""

    name = "recording"

    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []
        self.fail = False

    def tool_names(self) -> list[str]:
        return ["WebSearch", "mcp__knowledge__search_my_documents"]

    def mcp_server_names(self) -> list[str]:
        return ["playwright", "knowledge"]

    async def run(
        self, request: AgentRequest, *, search_documents: DocumentSearch, emitter: Any = None
    ) -> AgentOutcome:
        self.requests.append(request)
        recorder = TurnRecorder(emitter=emitter)
        recorder.session_id = request.resume_session_id or "sess-1"
        hits = await search_documents(request.question, 2)
        await recorder.add_step(NodeName.TOOL, {"label": "search", "hits": len(hits)})
        if self.fail:
            recorder.error = "The agent run failed: engine crashed"
        else:
            recorder.final_text = f"Answer using {len(hits)} passage(s)."
            if emitter:
                await emitter({"event": "token", "text": recorder.final_text})
        recorder.cost_usd = 0.01
        await recorder.finish()
        return AgentOutcome(
            text=recorder.final_text,
            session_id=recorder.session_id,
            steps=recorder.steps,
            is_error=recorder.final_text is None,
            error=recorder.error,
            cost_usd=recorder.cost_usd,
        )


@pytest.fixture
def agent_runner() -> RecordingRunner:
    return RecordingRunner()


async def agent_conversation(
    client: AsyncClient, headers: dict[str, str], **extra: Any
) -> dict[str, Any]:
    response = await client.post("/conversations", json={"mode": "agent", **extra}, headers=headers)
    assert response.status_code == 201, response.text
    return dict(response.json())


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    name = None
    for line in body.splitlines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1].strip())))
            name = None
    return events


class TestCreate:
    async def test_defaults_to_the_configured_model(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        conversation = await agent_conversation(client, auth_headers)
        assert conversation["mode"] == "agent"
        assert conversation["agent_model"] == "claude-opus-5-5"

    async def test_can_pick_fable(self, client: AsyncClient, auth_headers: dict[str, str]) -> None:
        conversation = await agent_conversation(client, auth_headers, model="claude-fable-5-2")
        assert conversation["agent_model"] == "claude-fable-5-2"

    async def test_unknown_model_is_rejected(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            "/conversations", json={"mode": "agent", "model": "gpt-9"}, headers=auth_headers
        )
        assert response.status_code == 400
        assert "not enabled" in response.json()["error"]["message"]

    async def test_tutor_remains_the_default(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post("/conversations", json={}, headers=auth_headers)
        assert response.json()["mode"] == "tutor"
        assert response.json()["agent_model"] is None


class TestTurns:
    async def test_turn_is_persisted_with_trace_and_session(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        agent_runner: RecordingRunner,
    ) -> None:
        await client.post(
            "/documents",
            files={"file": ("notes.md", b"The Battle of Plassey was in 1757.", "text/markdown")},
            headers=auth_headers,
        )
        conversation = await agent_conversation(client, auth_headers)

        response = await client.post(
            f"/conversations/{conversation['id']}/messages",
            json={"content": QUESTION},
            headers=auth_headers,
        )
        assert response.status_code == 201, response.text
        assistant = response.json()["assistant_message"]
        assert assistant["answer_status"] == "completed"
        assert assistant["content"] == "Answer using 1 passage(s)."
        assert [s["node"] for s in assistant["trace"]] == ["tool", "answer"]
        assert assistant["trace"][-1]["output"]["cost_usd"] == 0.01

        detail = (
            await client.get(f"/conversations/{conversation['id']}", headers=auth_headers)
        ).json()
        assert detail["title"] == QUESTION
        assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]

    async def test_second_turn_resumes_with_history(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        agent_runner: RecordingRunner,
    ) -> None:
        conversation = await agent_conversation(client, auth_headers, model="claude-fable-5-2")
        url = f"/conversations/{conversation['id']}/messages"
        await client.post(url, json={"content": QUESTION}, headers=auth_headers)
        await client.post(url, json={"content": "Who won it?"}, headers=auth_headers)

        first, second = agent_runner.requests
        assert first.resume_session_id is None
        assert first.model == "claude-fable-5-2"
        assert second.resume_session_id == "sess-1"
        assert [turn.content for turn in second.history][:1] == [QUESTION]

    async def test_failed_run_is_stored_as_failed(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        agent_runner: RecordingRunner,
    ) -> None:
        agent_runner.fail = True
        conversation = await agent_conversation(client, auth_headers)
        response = await client.post(
            f"/conversations/{conversation['id']}/messages",
            json={"content": QUESTION},
            headers=auth_headers,
        )
        assistant = response.json()["assistant_message"]
        assert assistant["answer_status"] == "failed"
        assert "engine crashed" in assistant["content"]
        assert assistant["error"] == "The agent run failed: engine crashed"

    async def test_tutor_conversations_still_use_the_tutor(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        api_conversation: dict[str, Any],
        agent_runner: RecordingRunner,
    ) -> None:
        response = await client.post(
            f"/conversations/{api_conversation['id']}/messages",
            json={"content": "What is 17 * 23?"},
            headers=auth_headers,
        )
        assert response.json()["assistant_message"]["answer_status"] == "verified"
        assert agent_runner.requests == []


class TestStreaming:
    async def test_stream_contract(self, client: AsyncClient, auth_headers: dict[str, str]) -> None:
        conversation = await agent_conversation(client, auth_headers)
        response = await client.post(
            f"/conversations/{conversation['id']}/messages/stream",
            json={"content": QUESTION},
            headers=auth_headers,
        )
        assert response.status_code == 200
        events = parse_sse(response.text)
        names = [name for name, _ in events]
        assert names[0] == "step"
        assert "token" in names
        assert names[-1] == "done"
        done = events[-1][1]
        assert done["answer_status"] == "completed"
        assert done["content"] == "Answer using 0 passage(s)."

    async def test_stream_for_someone_elses_conversation_is_404(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        other_auth_headers: dict[str, str],
    ) -> None:
        conversation = await agent_conversation(client, auth_headers)
        response = await client.post(
            f"/conversations/{conversation['id']}/messages/stream",
            json={"content": QUESTION},
            headers=other_auth_headers,
        )
        assert response.status_code == 404


class TestConfigAndHealth:
    async def test_agent_config(self, client: AsyncClient, auth_headers: dict[str, str]) -> None:
        response = await client.get("/agent/config", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "Lelouch"
        assert body["default_model"] == "claude-opus-5-5"
        assert "claude-fable-5-2" in body["models"]
        assert body["mcp_servers"] == ["playwright", "knowledge"]
        assert body["auth"] == "none"

    async def test_agent_config_needs_auth(self, client: AsyncClient) -> None:
        assert (await client.get("/agent/config")).status_code == 401

    async def test_health_reports_the_agent(self, client: AsyncClient) -> None:
        body = (await client.get("/health")).json()
        assert body["agent_backend"] == "fake"
        assert body["agent_model"] == "claude-opus-5-5"
        assert body["agent_auth"] == "none"
