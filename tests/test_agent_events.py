"""TurnRecorder: SDK messages -> trace steps, SSE events and the outcome."""

from __future__ import annotations

from typing import Any

from app.assistant.events import (
    TurnRecorder,
    friendly_tool_name,
    preview_result,
    summarise_input,
)
from app.models.enums import NodeName, StepStatus
from tests import agent_fakes as f


async def record(messages: list[Any]) -> tuple[TurnRecorder, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    recorder = TurnRecorder(emitter=emit)
    for message in messages:
        await recorder.handle(message)
    await recorder.finish()
    return recorder, events


class TestHelpers:
    def test_friendly_tool_name(self) -> None:
        assert friendly_tool_name("mcp__playwright__browser_click") == "playwright: browser_click"
        assert friendly_tool_name("WebSearch") == "WebSearch"
        assert friendly_tool_name("mcp__odd") == "mcp__odd"

    def test_summarise_input_prefers_meaningful_fields(self) -> None:
        assert summarise_input({"query": "  capital of France "}) == "capital of France"
        assert summarise_input({"url": "https://x.org", "prompt": "p"}) == "https://x.org"
        assert summarise_input({"ref": "e12"}) == '{"ref": "e12"}'
        assert len(summarise_input({"query": "x" * 1000})) == 300

    def test_preview_result_handles_every_shape(self) -> None:
        assert preview_result(None) == ""
        assert preview_result("abc") == "abc"
        assert preview_result([{"type": "text", "text": "a"}, {"type": "image"}, "b"]) == "a b"
        assert preview_result(42) == "42"


class TestTurnRecorder:
    async def test_a_full_research_turn(self) -> None:
        recorder, events = await record(f.research_script())

        nodes = [step.node for step in recorder.steps]
        assert nodes == [
            NodeName.AGENT_START,
            NodeName.THINKING,
            NodeName.TOOL,
            NodeName.TOOL,
            NodeName.ANSWER,
        ]
        assert [step.sequence for step in recorder.steps] == [1, 2, 3, 4, 5]
        search = recorder.steps[2].output
        assert search["tool"] == "WebSearch"
        assert search["input"] == "Battle of Plassey date"
        assert search["result_preview"] == "Plassey: 23 June 1757"
        assert recorder.steps[3].output["label"] == "knowledge: search_my_documents"

        assert recorder.final_text == "The battle was in 1757.\n\nSources: notes.md"
        assert recorder.session_id == f.SESSION
        assert recorder.cost_usd == 0.0421
        answer = recorder.steps[-1]
        assert answer.status is StepStatus.OK
        assert answer.output["cost_usd"] == 0.0421
        assert answer.output["num_turns"] == 3

        tokens = [e["text"] for e in events if e["event"] == "token"]
        assert "".join(tokens) == "The battle was in 1757."
        steps = [e for e in events if e["event"] == "step"]
        assert [e["node"] for e in steps] == [s.node.value for s in recorder.steps]
        assert steps[2]["label"] == "WebSearch"

    async def test_subagent_activity_is_traced_but_not_streamed(self) -> None:
        recorder, events = await record(
            [
                f.init(),
                f.assistant(f.tool_use("a1", "Task", description="research Plassey")),
                f.assistant(f.tool_use("s1", "WebSearch", query="Plassey"), parent="a1"),
                f.delta("subagent chatter", parent="a1"),
                f.assistant(f.text("subagent findings"), parent="a1"),
                f.tool_result("a1", "findings"),
                f.result("Main answer"),
            ]
        )
        assert recorder.steps[1].output["input"] == "research Plassey"
        assert recorder.steps[2].output["subagent"] is True
        assert not [e for e in events if e["event"] == "token"]
        assert recorder.final_text == "Main answer"

    async def test_failed_tool_marks_the_step(self) -> None:
        recorder, events = await record(
            [
                f.init(),
                f.assistant(f.tool_use("t1", "WebFetch", url="https://down.example")),
                f.tool_result("t1", "HTTP 503", is_error=True),
                f.result("Could not fetch, but here is what I know."),
            ]
        )
        fetch = recorder.steps[1]
        assert fetch.status is StepStatus.ERROR
        assert fetch.error == "HTTP 503"
        # Emitted once at start, once again when the failure is known.
        assert [e["status"] for e in events if e["event"] == "step" and e["sequence"] == 2] == [
            "ok",
            "error",
        ]

    async def test_mcp_connection_failure_is_flagged(self) -> None:
        recorder, _ = await record(
            [f.init({"playwright": "failed", "knowledge": "connected"}), f.result()]
        )
        start = recorder.steps[0]
        assert start.status is StepStatus.ERROR
        assert "playwright" in (start.error or "")
        assert start.output["mcp_servers"] == {"playwright": "failed", "knowledge": "connected"}

    async def test_error_result_fails_the_answer(self) -> None:
        recorder, _ = await record(
            [f.init(), f.result(None, is_error=True, errors=["max budget exceeded"])]
        )
        answer = recorder.steps[-1]
        assert recorder.is_error
        assert answer.status is StepStatus.ERROR
        assert "max budget exceeded" in (answer.error or "")

    async def test_falls_back_to_last_main_text_when_result_is_empty(self) -> None:
        recorder, _ = await record(
            [f.init(), f.assistant(f.text("first")), f.assistant(f.text("second")), f.result("")]
        )
        assert recorder.final_text == "second"

    async def test_no_answer_at_all_is_a_failure(self) -> None:
        recorder = TurnRecorder()
        await recorder.handle(f.init())
        answer = await recorder.finish()
        assert answer.status is StepStatus.ERROR
        assert answer.error == "The agent produced no answer"

    async def test_model_error_on_assistant_message_is_kept(self) -> None:
        message = f.assistant(f.text("partial"))
        message.error = "rate_limit"  # type: ignore[assignment]
        recorder, _ = await record([f.init(), message, f.result(None, is_error=True)])
        assert recorder.error == "Model error: rate_limit"

    async def test_irrelevant_messages_are_ignored(self) -> None:
        from claude_agent_sdk import StreamEvent, SystemMessage, UserMessage

        recorder, events = await record(
            [
                SystemMessage(subtype="compact_boundary", data={}),
                StreamEvent(uuid="u", session_id="s", event={"type": "message_start"}),
                UserMessage(content="plain text"),
                f.tool_result("unknown-id", "orphan"),
                object(),
                f.result("ok"),
            ]
        )
        assert [s.node for s in recorder.steps] == [NodeName.ANSWER]
        assert not [e for e in events if e["event"] == "token"]
