"""Agent runners, with the Claude engine replaced by a scripted ``query``."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import claude_agent_sdk
import pytest

from app.assistant.runner import (
    AgentRequest,
    ClaudeAgentRunner,
    FakeAgentRunner,
    _explain_failure,
    auth_mode,
    build_agent_runner,
    format_hits,
)
from app.config import Settings
from app.db.repository import ChunkHit
from app.models.domain import HistoryTurn
from app.models.enums import MessageRole, NodeName, StepStatus
from tests import agent_fakes as f

HIT = ChunkHit(
    document_id=uuid.uuid4(), filename="notes.md", chunk_index=0, content="Plassey 1757", score=0.9
)


async def no_docs(query: str, top_k: int | None) -> list[ChunkHit]:
    return []


async def one_doc(query: str, top_k: int | None) -> list[ChunkHit]:
    return [HIT]


class ScriptedQuery:
    """Stands in for ``claude_agent_sdk.query``; one script per call."""

    def __init__(self, *scripts: list[Any] | BaseException) -> None:
        self.scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, prompt: str, options: Any) -> AsyncIterator[Any]:
        self.calls.append({"prompt": prompt, "options": options})
        script = self.scripts.pop(0)

        async def stream() -> AsyncIterator[Any]:
            if isinstance(script, BaseException):
                raise script
            for item in script:
                if isinstance(item, BaseException):
                    raise item
                yield item

        return stream()


@pytest.fixture
def agent_settings(settings: Settings, tmp_path: Any) -> Settings:
    return settings.model_copy(
        update={"agent_backend": "claude_sdk", "agent_workspace_dir": str(tmp_path / "ws")}
    )


def request(**overrides: Any) -> AgentRequest:
    values: dict[str, Any] = {
        "question": "When was Plassey?",
        "model": "claude-opus-5-5",
        "user_id": uuid.uuid4(),
    }
    values.update(overrides)
    return AgentRequest(**values)


def install(monkeypatch: pytest.MonkeyPatch, fake: ScriptedQuery) -> ScriptedQuery:
    monkeypatch.setattr(claude_agent_sdk, "query", fake)
    return fake


class TestOptions:
    def test_locked_down_toolset(self, agent_settings: Settings) -> None:
        runner = ClaudeAgentRunner(agent_settings)
        options = runner.build_options(request(), search_documents=no_docs, resume=False)

        assert options.permission_mode == "dontAsk"
        assert options.strict_mcp_config is True
        assert options.setting_sources == []
        assert "Bash" not in options.tools and "Write" not in options.tools
        assert {"WebSearch", "WebFetch", "Agent", "TaskCreate"} <= set(options.tools)
        assert "mcp__knowledge" in options.allowed_tools
        assert set(options.mcp_servers) == {"knowledge"}
        assert options.mcp_servers["knowledge"]["type"] == "sdk"
        assert options.agents and "researcher" in options.agents
        assert options.thinking == {"type": "adaptive"}
        assert options.effort == "high"
        assert options.max_budget_usd == agent_settings.agent_max_budget_usd
        assert options.include_partial_messages is True
        assert options.resume is None
        assert "Lelouch" in options.system_prompt

    def test_workspace_is_per_user(self, agent_settings: Settings) -> None:
        runner = ClaudeAgentRunner(agent_settings)
        user_id = uuid.uuid4()
        options = runner.build_options(
            request(user_id=user_id), search_documents=no_docs, resume=False
        )
        assert str(options.cwd).endswith(str(user_id))

    def test_playwright_and_user_servers_are_allowed(
        self, agent_settings: Settings, tmp_path: Any
    ) -> None:
        config = tmp_path / "mcp.json"
        config.write_text('{"mcpServers": {"notion": {"type": "http", "url": "https://n"}}}')
        runner = ClaudeAgentRunner(
            agent_settings.model_copy(
                update={"agent_enable_playwright": True, "agent_mcp_config_path": str(config)}
            )
        )
        options = runner.build_options(request(), search_documents=no_docs, resume=False)
        assert {"mcp__playwright", "mcp__notion", "mcp__knowledge"} <= set(options.allowed_tools)
        assert runner.mcp_server_names() == ["playwright", "notion", "knowledge"]

    def test_web_can_be_switched_off(self, agent_settings: Settings) -> None:
        runner = ClaudeAgentRunner(agent_settings.model_copy(update={"agent_enable_web": False}))
        options = runner.build_options(request(), search_documents=no_docs, resume=False)
        assert "WebSearch" not in options.tools
        assert options.agents is None

    def test_thinking_can_be_disabled(self, agent_settings: Settings) -> None:
        runner = ClaudeAgentRunner(agent_settings.model_copy(update={"agent_thinking": "disabled"}))
        options = runner.build_options(request(), search_documents=no_docs, resume=False)
        assert options.thinking == {"type": "disabled"}


class TestSearchTool:
    async def test_returns_formatted_hits(self) -> None:
        tool = ClaudeAgentRunner.search_tool(one_doc)
        assert tool.name == "search_my_documents"
        response = await tool.handler({"query": "Plassey"})
        assert "notes.md" in response["content"][0]["text"]
        assert "Plassey 1757" in response["content"][0]["text"]

    async def test_reports_failures_to_the_model(self) -> None:
        async def broken(query: str, top_k: int | None) -> list[ChunkHit]:
            raise RuntimeError("db down")

        response = await ClaudeAgentRunner.search_tool(broken).handler({"query": "x"})
        assert response["is_error"] is True
        assert "db down" in response["content"][0]["text"]

    def test_format_hits(self) -> None:
        assert format_hits([]) == "No matching passages in the user's documents."
        assert format_hits([HIT, HIT]).count("notes.md") == 2


class TestRun:
    async def test_successful_turn(
        self, agent_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = install(monkeypatch, ScriptedQuery(f.research_script()))
        events: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            events.append(event)

        outcome = await ClaudeAgentRunner(agent_settings).run(
            request(), search_documents=no_docs, emitter=emit
        )
        assert not outcome.is_error
        assert outcome.text and outcome.text.startswith("The battle was in 1757.")
        assert outcome.session_id == f.SESSION
        assert outcome.cost_usd == 0.0421
        assert fake.calls[0]["prompt"] == "When was Plassey?"
        assert any(e["event"] == "token" for e in events)

    async def test_first_turn_replays_history(
        self, agent_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = install(monkeypatch, ScriptedQuery([f.init(), f.result("ok")]))
        history = [
            HistoryTurn(role=MessageRole.USER, content="Earlier question"),
            HistoryTurn(role=MessageRole.ASSISTANT, content="Earlier answer"),
        ]
        await ClaudeAgentRunner(agent_settings).run(
            request(history=history), search_documents=no_docs
        )
        prompt = fake.calls[0]["prompt"]
        assert "Earlier question" in prompt and "Earlier answer" in prompt
        assert prompt.endswith("When was Plassey?")

    async def test_resumes_the_sdk_session(
        self, agent_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = install(monkeypatch, ScriptedQuery([f.init(), f.result("ok")]))
        history = [HistoryTurn(role=MessageRole.USER, content="Earlier question")]
        await ClaudeAgentRunner(agent_settings).run(
            request(resume_session_id="old", history=history), search_documents=no_docs
        )
        assert fake.calls[0]["options"].resume == "old"
        # The resumed session already has the context; don't send it twice.
        assert fake.calls[0]["prompt"] == "When was Plassey?"

    async def test_lost_session_retries_fresh(
        self, agent_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = install(
            monkeypatch,
            ScriptedQuery(
                RuntimeError("No conversation found with session ID: old"),
                [f.init(session_id="new"), f.result("recovered", session_id="new")],
            ),
        )
        history = [HistoryTurn(role=MessageRole.USER, content="Earlier question")]
        outcome = await ClaudeAgentRunner(agent_settings).run(
            request(resume_session_id="old", history=history), search_documents=no_docs
        )
        assert outcome.text == "recovered"
        assert outcome.session_id == "new"
        assert fake.calls[1]["options"].resume is None
        assert "Earlier question" in fake.calls[1]["prompt"]

    async def test_failure_without_resume_is_reported_not_raised(
        self, agent_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, ScriptedQuery(RuntimeError("Authentication error")))
        outcome = await ClaudeAgentRunner(agent_settings).run(request(), search_documents=no_docs)
        assert outcome.is_error
        assert "not signed in" in (outcome.error or "")
        assert outcome.steps[-1].node is NodeName.ANSWER
        assert outcome.steps[-1].status is StepStatus.ERROR

    async def test_timeout(self, agent_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        class Slow:
            def __call__(self, *, prompt: str, options: Any) -> AsyncIterator[Any]:
                async def stream() -> AsyncIterator[Any]:
                    yield f.init()
                    await asyncio.sleep(5)
                    yield f.result()

                return stream()

        monkeypatch.setattr(claude_agent_sdk, "query", Slow())
        runner = ClaudeAgentRunner(
            agent_settings.model_copy(update={"agent_timeout_seconds": 0.05})
        )
        outcome = await runner.run(request(), search_documents=no_docs)
        assert outcome.is_error
        assert "did not finish within" in (outcome.error or "")

    def test_failure_messages(self) -> None:
        assert "not found" in _explain_failure(claude_agent_sdk.CLINotFoundError("missing"))
        assert _explain_failure(RuntimeError("boom")) == "The agent run failed: boom"


class TestFakeRunner:
    async def test_uses_the_document_search(self) -> None:
        outcome = await FakeAgentRunner().run(request(), search_documents=one_doc)
        assert "notes.md" in (outcome.text or "")
        assert [s.node for s in outcome.steps] == [
            NodeName.AGENT_START,
            NodeName.THINKING,
            NodeName.TOOL,
            NodeName.ANSWER,
        ]
        assert outcome.session_id and outcome.session_id.startswith("fake-")

    async def test_nothing_found(self) -> None:
        outcome = await FakeAgentRunner().run(request(), search_documents=no_docs)
        assert "could not find" in (outcome.text or "")


class TestFactoryAndAuth:
    def test_factory(self, settings: Settings) -> None:
        assert isinstance(build_agent_runner(settings), FakeAgentRunner)
        real = build_agent_runner(settings.model_copy(update={"agent_backend": "claude_sdk"}))
        assert isinstance(real, ClaudeAgentRunner)

    def test_auth_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert auth_mode() == "claude_login"
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert auth_mode() == "api_key"
