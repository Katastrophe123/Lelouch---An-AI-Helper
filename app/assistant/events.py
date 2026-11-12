"""Translate Claude Agent SDK messages into trace steps and SSE events.

The SDK yields a stream of typed messages (system init, assistant content
blocks, tool results, partial text deltas, a final result). ``TurnRecorder``
folds them into the same shapes the tutor graph already produces - a list of
:class:`StepRecord` for the stored trace, and ``step`` / ``token`` events for
the streaming endpoint - so storage, the API and the UI need no agent-specific
contract.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    ServerToolUseBlock,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from app.models.domain import StepRecord
from app.models.enums import NodeName, RoutingDecision, StepStatus, StreamEventType

Emitter = Callable[[dict[str, Any]], Awaitable[None]]

PREVIEW_CHARS = 300

#: The input field that best summarises each tool call, for the trace label.
_SUMMARY_FIELDS = ("query", "url", "description", "prompt", "element", "text", "pattern")


def summarise_input(tool_input: dict[str, Any]) -> str:
    for key in _SUMMARY_FIELDS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:PREVIEW_CHARS]
    try:
        return json.dumps(tool_input, ensure_ascii=False, default=str)[:PREVIEW_CHARS]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(tool_input)[:PREVIEW_CHARS]


def preview_result(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:PREVIEW_CHARS]
    if isinstance(content, list):
        texts = [
            str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
        ]
        return " ".join(t for t in texts if t)[:PREVIEW_CHARS]
    return str(content)[:PREVIEW_CHARS]


def friendly_tool_name(name: str) -> str:
    """``mcp__playwright__browser_click`` -> ``playwright: browser_click``."""
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return f"{parts[1]}: {parts[2]}"
    return name


@dataclass
class TurnRecorder:
    """Accumulates one agent turn; feed it every SDK message in order."""

    emitter: Emitter | None = None
    steps: list[StepRecord] = field(default_factory=list)
    session_id: str | None = None
    final_text: str | None = None
    is_error: bool = False
    error: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    usage: dict[str, Any] | None = None
    streamed_chars: int = 0
    _main_texts: list[str] = field(default_factory=list)
    _tool_steps: dict[str, int] = field(default_factory=dict)
    _started: float = field(default_factory=perf_counter)
    _last_mark: float = field(default_factory=perf_counter)

    # --- plumbing ---------------------------------------------------------
    def _elapsed(self) -> int:
        now = perf_counter()
        elapsed = max(0, int((now - self._last_mark) * 1000))
        self._last_mark = now
        return elapsed

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.emitter is not None:
            await self.emitter(event)

    async def _emit_step(self, record: StepRecord) -> None:
        await self._emit(
            {
                "event": StreamEventType.STEP.value,
                "node": record.node.value,
                "sequence": record.sequence,
                "status": record.status.value,
                "decision": record.routing_decision.value if record.routing_decision else None,
                "latency_ms": record.latency_ms,
                "label": record.output.get("label"),
                # Already truncated previews, so clients can describe the step
                # live ("Searched the web: <query>") without waiting for the trace.
                "output": record.output,
                "error": record.error,
            }
        )

    async def add_step(
        self,
        node: NodeName,
        output: dict[str, Any],
        *,
        status: StepStatus = StepStatus.OK,
        error: str | None = None,
        decision: RoutingDecision | None = None,
    ) -> StepRecord:
        record = StepRecord(
            sequence=len(self.steps) + 1,
            node=node,
            status=status,
            latency_ms=self._elapsed(),
            output=output,
            routing_decision=decision,
            error=error,
        )
        self.steps.append(record)
        await self._emit_step(record)
        return record

    # --- message handlers -------------------------------------------------
    async def handle(self, message: Any) -> None:
        if isinstance(message, StreamEvent):
            await self._on_stream_event(message)
        elif isinstance(message, SystemMessage):
            await self._on_system(message)
        elif isinstance(message, AssistantMessage):
            await self._on_assistant(message)
        elif isinstance(message, UserMessage):
            await self._on_user(message)
        elif isinstance(message, ResultMessage):
            await self._on_result(message)

    async def _on_stream_event(self, message: StreamEvent) -> None:
        # Only the main agent's prose is streamed; subagents work silently.
        if message.parent_tool_use_id is not None:
            return
        event = message.event or {}
        if event.get("type") != "content_block_delta":
            return
        delta = event.get("delta") or {}
        if delta.get("type") == "text_delta" and delta.get("text"):
            self.streamed_chars += len(delta["text"])
            await self._emit({"event": StreamEventType.TOKEN.value, "text": delta["text"]})

    async def _on_system(self, message: SystemMessage) -> None:
        if message.subtype != "init":
            return
        data = message.data or {}
        self.session_id = data.get("session_id") or self.session_id
        servers = data.get("mcp_servers") or []
        failed = [
            s.get("name", "?")
            for s in servers
            if isinstance(s, dict) and s.get("status") not in (None, "connected")
        ]
        output = {
            "label": "Agent started",
            "model": data.get("model"),
            "tools": len(data.get("tools") or []),
            "mcp_servers": {
                s.get("name", "?"): s.get("status") for s in servers if isinstance(s, dict)
            },
        }
        await self.add_step(
            NodeName.AGENT_START,
            output,
            status=StepStatus.ERROR if failed else StepStatus.OK,
            error=f"MCP server(s) failed to connect: {', '.join(failed)}" if failed else None,
        )

    async def _on_assistant(self, message: AssistantMessage) -> None:
        subagent = message.parent_tool_use_id is not None
        if message.error:
            self.error = f"Model error: {message.error}"
        for block in message.content:
            if isinstance(block, ThinkingBlock) and block.thinking.strip():
                await self.add_step(
                    NodeName.THINKING,
                    {
                        "label": "Thinking",
                        "preview": block.thinking.strip()[:PREVIEW_CHARS],
                        "subagent": subagent,
                    },
                )
            elif isinstance(block, ToolUseBlock | ServerToolUseBlock):
                await self.add_step(
                    NodeName.TOOL,
                    {
                        "label": friendly_tool_name(block.name),
                        "tool": block.name,
                        "input": summarise_input(block.input or {}),
                        "subagent": subagent,
                    },
                )
                self._tool_steps[block.id] = len(self.steps) - 1
            elif isinstance(block, TextBlock) and not subagent and block.text.strip():
                self._main_texts.append(block.text.strip())

    async def _on_user(self, message: UserMessage) -> None:
        if not isinstance(message.content, list):
            return
        for block in message.content:
            if not isinstance(block, ToolResultBlock):
                continue
            index = self._tool_steps.get(block.tool_use_id)
            if index is None:
                continue
            record = self.steps[index]
            record.output["result_preview"] = preview_result(block.content)
            if block.is_error:
                record.status = StepStatus.ERROR
                record.error = preview_result(block.content) or "Tool call failed"
                await self._emit_step(record)

    async def _on_result(self, message: ResultMessage) -> None:
        self.session_id = message.session_id or self.session_id
        self.cost_usd = message.total_cost_usd
        self.num_turns = message.num_turns
        self.usage = message.usage
        self.is_error = bool(message.is_error)
        if message.result and message.result.strip():
            self.final_text = message.result.strip()
        elif self._main_texts:
            self.final_text = self._main_texts[-1]
        if self.is_error:
            detail = "; ".join(str(e) for e in (message.errors or [])) or message.subtype
            self.error = self.error or f"Agent run ended with an error ({detail})"

    async def finish(self) -> StepRecord:
        """Append the closing ``answer`` step, carrying cost and usage."""
        if self.final_text is None and self._main_texts:
            self.final_text = self._main_texts[-1]
        failed = self.is_error or not self.final_text
        return await self.add_step(
            NodeName.ANSWER,
            {
                "label": "Answer" if not failed else "Run failed",
                "answer_chars": len(self.final_text or ""),
                "num_turns": self.num_turns,
                "cost_usd": self.cost_usd,
                "usage": self.usage,
                "total_ms": max(0, int((perf_counter() - self._started) * 1000)),
            },
            status=StepStatus.ERROR if failed else StepStatus.OK,
            error=(self.error or "The agent produced no answer") if failed else None,
            decision=RoutingDecision.END,
        )
