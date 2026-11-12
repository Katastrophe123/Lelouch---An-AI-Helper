"""Builders for Claude Agent SDK messages, so tests can script the engine."""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

SESSION = "sess-123"


def init(servers: dict[str, str] | None = None, session_id: str = SESSION) -> SystemMessage:
    servers = (
        servers if servers is not None else {"playwright": "connected", "knowledge": "connected"}
    )
    return SystemMessage(
        subtype="init",
        data={
            "session_id": session_id,
            "model": "claude-opus-5-5",
            "tools": ["WebSearch", "WebFetch"],
            "mcp_servers": [{"name": n, "status": s} for n, s in servers.items()],
        },
    )


def assistant(*blocks: Any, parent: str | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=list(blocks), model="claude-opus-5-5", parent_tool_use_id=parent
    )


def thinking(text: str) -> ThinkingBlock:
    return ThinkingBlock(thinking=text, signature="sig")


def tool_use(tool_id: str, name: str, **tool_input: Any) -> ToolUseBlock:
    return ToolUseBlock(id=tool_id, name=name, input=tool_input)


def tool_result(tool_id: str, content: Any, *, is_error: bool = False) -> UserMessage:
    return UserMessage(
        content=[ToolResultBlock(tool_use_id=tool_id, content=content, is_error=is_error)]
    )


def text(value: str) -> TextBlock:
    return TextBlock(text=value)


def delta(value: str, parent: str | None = None) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id=SESSION,
        event={
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": value},
        },
        parent_tool_use_id=parent,
    )


def result(
    value: str | None = "Final answer.",
    *,
    is_error: bool = False,
    session_id: str = SESSION,
    errors: list[str] | None = None,
) -> ResultMessage:
    return ResultMessage(
        subtype="error_during_execution" if is_error else "success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=is_error,
        num_turns=3,
        session_id=session_id,
        total_cost_usd=0.0421,
        usage={"input_tokens": 100, "output_tokens": 50},
        result=value,
        errors=errors,
    )


def research_script() -> list[Any]:
    """A typical turn: think, search the web, read the user's docs, answer."""
    return [
        init(),
        assistant(thinking("I should search the web and the user's notes.")),
        assistant(tool_use("t1", "WebSearch", query="Battle of Plassey date")),
        tool_result("t1", "Plassey: 23 June 1757"),
        assistant(tool_use("t2", "mcp__knowledge__search_my_documents", query="Plassey")),
        tool_result("t2", [{"type": "text", "text": "[1] notes.md ... 1757"}]),
        delta("The battle "),
        delta("was in 1757."),
        assistant(text("The battle was in 1757.\n\nSources: notes.md")),
        result("The battle was in 1757.\n\nSources: notes.md"),
    ]
