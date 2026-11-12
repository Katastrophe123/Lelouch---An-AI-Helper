"""Agent runners: the real Claude Agent SDK, and an offline fake.

Authentication
--------------
``ClaudeAgentRunner`` launches the Claude Code engine bundled with the
``claude-agent-sdk`` package. It authenticates exactly as the ``claude`` CLI
on this machine does:

* signed in with your Claude subscription (``claude`` then ``/login``): usage
  is billed to your plan's monthly **Agent SDK credit**, for your own use;
* ``ANTHROPIC_API_KEY`` set in the environment: billed to that API key. This
  is the route for any version other people use.

This code never reads, copies or forwards a credential; it only starts the
official engine, which handles sign-in itself.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.assistant.events import Emitter, TurnRecorder
from app.assistant.mcp_config import build_external_servers
from app.assistant.prompts import (
    RESEARCHER_PROMPT,
    build_prompt_with_history,
    build_system_prompt,
)
from app.config import Settings
from app.db.repository import ChunkHit
from app.logging_config import get_logger
from app.models.domain import HistoryTurn, StepRecord
from app.models.enums import NodeName

logger = get_logger(__name__)

DocumentSearch = Callable[[str, int | None], Awaitable[list[ChunkHit]]]

KNOWLEDGE_SERVER = "knowledge"
SEARCH_TOOL = "search_my_documents"


def auth_mode() -> str:
    """Which credential the engine will use; reported by /health."""
    return "api_key" if os.environ.get("ANTHROPIC_API_KEY") else "claude_login"


def format_hits(hits: list[ChunkHit]) -> str:
    if not hits:
        return "No matching passages in the user's documents."
    blocks = [
        f"[{n}] {hit.filename} (chunk {hit.chunk_index}, similarity {hit.score:.2f})\n{hit.content}"
        for n, hit in enumerate(hits, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


@dataclass(frozen=True, slots=True)
class AgentRequest:
    question: str
    model: str
    user_id: uuid.UUID
    history: list[HistoryTurn] = field(default_factory=list)
    resume_session_id: str | None = None


@dataclass(slots=True)
class AgentOutcome:
    text: str | None
    session_id: str | None
    steps: list[StepRecord]
    is_error: bool = False
    error: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None


class AgentRunner(ABC):
    name: str = "agent"

    @abstractmethod
    async def run(
        self,
        request: AgentRequest,
        *,
        search_documents: DocumentSearch,
        emitter: Emitter | None = None,
    ) -> AgentOutcome:
        """Run one turn. Must not raise for model or tool failures."""

    def tool_names(self) -> list[str]:
        return []

    def mcp_server_names(self) -> list[str]:
        return []


def _outcome(recorder: TurnRecorder) -> AgentOutcome:
    return AgentOutcome(
        text=recorder.final_text,
        session_id=recorder.session_id,
        steps=recorder.steps,
        is_error=recorder.is_error or not recorder.final_text,
        error=recorder.error,
        cost_usd=recorder.cost_usd,
        num_turns=recorder.num_turns,
    )


# --------------------------------------------------------------------------- #
# Real runner
# --------------------------------------------------------------------------- #
class ClaudeAgentRunner(AgentRunner):
    name = "claude_sdk"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Parsed once at startup, so a broken mcp_servers.json fails loudly.
        self._external_servers = build_external_servers(settings)

    def builtin_tools(self) -> list[str]:
        # Planning tools: TaskCreate/TaskUpdate/... in current engines, TodoWrite
        # in older ones. Names the engine does not know are simply ignored.
        tools = ["TaskCreate", "TaskUpdate", "TaskList", "TaskGet", "TodoWrite"]
        if self._settings.agent_enable_web:
            tools += ["WebSearch", "WebFetch", "Agent"]
        return tools

    def tool_names(self) -> list[str]:
        return [*self.builtin_tools(), f"mcp__{KNOWLEDGE_SERVER}__{SEARCH_TOOL}"]

    def mcp_server_names(self) -> list[str]:
        return [*self._external_servers, KNOWLEDGE_SERVER]

    def _workspace(self, user_id: uuid.UUID) -> Path:
        path = Path(self._settings.agent_workspace_dir).resolve() / str(user_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _knowledge_server(self, search_documents: DocumentSearch) -> Any:
        from claude_agent_sdk import create_sdk_mcp_server

        return create_sdk_mcp_server(KNOWLEDGE_SERVER, tools=[self.search_tool(search_documents)])

    @staticmethod
    def search_tool(search_documents: DocumentSearch) -> Any:
        """The ``search_my_documents`` tool, bound to one user's search."""
        from claude_agent_sdk import tool

        @tool(
            SEARCH_TOOL,
            "Search the user's own uploaded documents (notes, PDFs, Word files, "
            "data) and return the most relevant passages with their file names. "
            "Use it whenever the question might involve the user's own material.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                    "top_k": {
                        "type": "integer",
                        "description": "How many passages to return (default 6)",
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
            },
        )
        async def search_my_documents(args: dict[str, Any]) -> dict[str, Any]:
            try:
                hits = await search_documents(str(args["query"]), args.get("top_k"))
            except Exception as exc:  # noqa: BLE001 - report to the model, don't crash
                logger.warning("knowledge_tool_failed", error=str(exc))
                return {
                    "content": [{"type": "text", "text": f"Document search failed: {exc}"}],
                    "is_error": True,
                }
            return {"content": [{"type": "text", "text": format_hits(hits)}]}

        return search_my_documents

    def build_options(
        self, request: AgentRequest, *, search_documents: DocumentSearch, resume: bool
    ) -> Any:
        from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions

        settings = self._settings
        servers: dict[str, Any] = {
            **self._external_servers,
            KNOWLEDGE_SERVER: self._knowledge_server(search_documents),
        }
        builtins = self.builtin_tools()
        # "dontAsk" denies anything not pre-approved here: no shell, no file
        # edits on your machine, nothing the list below does not name.
        allowed = [*builtins, "Task", *(f"mcp__{name}" for name in servers)]
        agents = None
        if settings.agent_enable_web:
            agents = {
                "researcher": AgentDefinition(
                    description=(
                        "Deep web researcher. Give it one focused sub-question; it "
                        "searches, reads primary sources and returns cited findings."
                    ),
                    prompt=RESEARCHER_PROMPT,
                    tools=["WebSearch", "WebFetch"],
                )
            }
        thinking: dict[str, Any] = (
            {"type": "adaptive"} if settings.agent_thinking == "adaptive" else {"type": "disabled"}
        )
        return ClaudeAgentOptions(
            model=request.model,
            system_prompt=build_system_prompt(
                name=settings.agent_name, today=datetime.now(tz=UTC).date().isoformat()
            ),
            tools=builtins,
            allowed_tools=allowed,
            permission_mode="dontAsk",
            mcp_servers=servers,
            strict_mcp_config=True,
            setting_sources=[],
            agents=agents,
            max_turns=settings.agent_max_turns,
            max_budget_usd=settings.agent_max_budget_usd,
            effort=settings.agent_effort,
            thinking=thinking,  # type: ignore[arg-type]
            include_partial_messages=True,
            cwd=self._workspace(request.user_id),
            resume=request.resume_session_id if resume else None,
            cli_path=settings.agent_cli_path,
            stderr=lambda line: logger.debug("claude_cli_stderr", line=line[:500]),
        )

    async def _attempt(
        self,
        request: AgentRequest,
        recorder: TurnRecorder,
        *,
        search_documents: DocumentSearch,
        resume: bool,
    ) -> None:
        from claude_agent_sdk import query

        prompt = (
            request.question
            if resume
            else build_prompt_with_history(request.question, request.history)
        )
        options = self.build_options(request, search_documents=search_documents, resume=resume)
        async with asyncio.timeout(self._settings.agent_timeout_seconds):
            async for message in query(prompt=prompt, options=options):
                await recorder.handle(message)

    async def run(
        self,
        request: AgentRequest,
        *,
        search_documents: DocumentSearch,
        emitter: Emitter | None = None,
    ) -> AgentOutcome:
        recorder = TurnRecorder(emitter=emitter)
        resume = request.resume_session_id is not None
        try:
            try:
                await self._attempt(
                    request, recorder, search_documents=search_documents, resume=resume
                )
            except Exception as exc:
                # A resumed session can vanish (e.g. ~/.claude was cleared).
                # Retry once from scratch, replaying the stored history instead.
                if not resume or recorder.num_turns:
                    raise
                logger.warning("agent_resume_failed_retrying_fresh", error=str(exc))
                recorder.session_id = None
                await self._attempt(
                    request, recorder, search_documents=search_documents, resume=False
                )
        except TimeoutError:
            recorder.error = (
                f"The agent did not finish within {self._settings.agent_timeout_seconds:g}s"
            )
        except Exception as exc:  # noqa: BLE001 - a turn must always produce a message
            logger.exception("agent_run_failed", error=str(exc))
            recorder.error = _explain_failure(exc)
        await recorder.finish()
        return _outcome(recorder)


def _explain_failure(exc: Exception) -> str:
    from claude_agent_sdk import CLINotFoundError

    if isinstance(exc, CLINotFoundError):
        return "The Claude engine was not found. Reinstall claude-agent-sdk or set AGENT_CLI_PATH."
    text = str(exc)
    lowered = text.lower()
    if "login" in lowered or "auth" in lowered or "401" in lowered:
        return (
            "The Claude engine is not signed in. Run `claude` in a terminal and use "
            "/login with your Claude account (or set ANTHROPIC_API_KEY). "
            f"Detail: {text[:300]}"
        )
    return f"The agent run failed: {text[:500]}"


# --------------------------------------------------------------------------- #
# Offline runner
# --------------------------------------------------------------------------- #
class FakeAgentRunner(AgentRunner):
    """Deterministic stand-in: no network, no CLI, no credentials.

    It still exercises the real plumbing - it calls the actual document search,
    streams tokens and records steps through :class:`TurnRecorder` - so tests
    and ``AGENT_BACKEND=fake`` demos cover everything except the model itself.
    """

    name = "fake"

    def tool_names(self) -> list[str]:
        return [f"mcp__{KNOWLEDGE_SERVER}__{SEARCH_TOOL}"]

    def mcp_server_names(self) -> list[str]:
        return [KNOWLEDGE_SERVER]

    async def run(
        self,
        request: AgentRequest,
        *,
        search_documents: DocumentSearch,
        emitter: Emitter | None = None,
    ) -> AgentOutcome:
        recorder = TurnRecorder(emitter=emitter)
        recorder.session_id = request.resume_session_id or f"fake-{uuid.uuid4()}"
        await recorder.add_step(
            NodeName.AGENT_START,
            {
                "label": "Agent started",
                "model": request.model,
                "resumed": bool(request.resume_session_id),
            },
        )
        await recorder.add_step(
            NodeName.THINKING,
            {
                "label": "Thinking",
                "preview": "Check the user's documents first.",
                "subagent": False,
            },
        )
        record = await recorder.add_step(
            NodeName.TOOL,
            {
                "label": f"{KNOWLEDGE_SERVER}: {SEARCH_TOOL}",
                "tool": f"mcp__{KNOWLEDGE_SERVER}__{SEARCH_TOOL}",
                "input": request.question[:300],
                "subagent": False,
            },
        )
        hits = await search_documents(request.question, 3)
        record.output["result_preview"] = format_hits(hits)[:300]

        if hits:
            sources = ", ".join(dict.fromkeys(hit.filename for hit in hits))
            text = (
                f"(offline demo agent) I found {len(hits)} relevant passage(s) in your "
                f"documents.\n\n> {hits[0].content[:400]}\n\n**Sources:** {sources}"
            )
        else:
            text = (
                "(offline demo agent) I could not find anything about that in your "
                "documents. Set AGENT_BACKEND=claude_sdk to use the real agent."
            )
        if recorder.emitter is not None:
            for start in range(0, len(text), 24):
                await recorder.emitter({"event": "token", "text": text[start : start + 24]})
        recorder.final_text = text
        recorder.num_turns = 1
        recorder.cost_usd = 0.0
        await recorder.finish()
        return _outcome(recorder)


def build_agent_runner(settings: Settings) -> AgentRunner:
    if settings.agent_backend == "fake":
        return FakeAgentRunner()
    return ClaudeAgentRunner(settings)
