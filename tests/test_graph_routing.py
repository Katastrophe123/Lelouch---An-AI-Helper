"""End-to-end coverage of every conditional edge in the tutoring graph.

Each test drives the compiled graph with a scripted LLM so the branch taken is
fully determined, then asserts on the terminal state *and* on the recorded
routing decisions - the trace is what gets persisted with the message.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.graph import (
    route_after_explain,
    route_after_self_check,
    route_after_solve,
    route_after_understand,
)
from app.agents.state import initial_state
from app.llm.base import LLMPermanentError, LLMTransientError
from app.models.domain import HistoryTurn
from app.models.enums import (
    AnswerStatus,
    FallbackReason,
    MessageRole,
    NodeName,
    RoutingDecision,
)
from tests.fakes import (
    EXPLANATION,
    ScriptedLLMClient,
    build_self_check,
    build_solution,
    build_understanding,
    happy_script,
)

QUESTION = "A ball is dropped from 20 m. How long until it hits the ground?"


async def run_graph(
    graph: Any, question: str = QUESTION, history: list[HistoryTurn] | None = None
) -> dict[str, Any]:
    return await graph.ainvoke(
        initial_state(
            run_id="test-run",
            conversation_id="test-conversation",
            question=question,
            history=history,
        ),
        config={"recursion_limit": 30},
    )


def nodes_visited(state: dict[str, Any]) -> list[str]:
    return [step["node"] for step in state["trace"]]


def decisions(state: dict[str, Any]) -> list[str | None]:
    return [step["routing_decision"] for step in state["trace"]]


class TestRouterFunctions:
    """The routers in isolation: pure functions of ``last_decision``."""

    def test_understand_continues_to_solve(self) -> None:
        state = {"last_decision": RoutingDecision.TO_SOLVE.value}
        assert route_after_understand(state) == NodeName.SOLVE.value  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "decision",
        [
            RoutingDecision.TO_FALLBACK_NOT_ANSWERABLE,
            RoutingDecision.TO_FALLBACK_NODE_ERROR,
            RoutingDecision.TO_FALLBACK_STEP_LIMIT,
        ],
    )
    def test_understand_falls_back_on_any_fallback_decision(
        self, decision: RoutingDecision
    ) -> None:
        state = {"last_decision": decision.value}
        assert route_after_understand(state) == NodeName.FALLBACK.value  # type: ignore[arg-type]

    def test_solve_router_branches(self) -> None:
        ok = {"last_decision": RoutingDecision.TO_SELF_CHECK.value}
        bad = {"last_decision": RoutingDecision.TO_FALLBACK_NODE_ERROR.value}
        assert route_after_solve(ok) == NodeName.SELF_CHECK.value  # type: ignore[arg-type]
        assert route_after_solve(bad) == NodeName.FALLBACK.value  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("decision", "expected"),
        [
            (RoutingDecision.TO_EXPLAIN, NodeName.EXPLAIN),
            (RoutingDecision.RETRY_SOLVE, NodeName.SOLVE),
            (RoutingDecision.TO_FALLBACK_CHECK_FAILED, NodeName.FALLBACK),
        ],
    )
    def test_self_check_router_is_three_way(
        self, decision: RoutingDecision, expected: NodeName
    ) -> None:
        state = {"last_decision": decision.value}
        assert route_after_self_check(state) == expected.value  # type: ignore[arg-type]

    def test_unknown_decision_fails_safe_to_fallback(self) -> None:
        assert route_after_self_check({"last_decision": "something-new"}) == (  # type: ignore[arg-type]
            NodeName.FALLBACK.value
        )

    def test_explain_router_ends_or_falls_back(self) -> None:
        assert route_after_explain({"last_decision": RoutingDecision.END.value}) == "__end__"  # type: ignore[arg-type]
        assert (
            route_after_explain(
                {"last_decision": RoutingDecision.TO_FALLBACK_NODE_ERROR.value}  # type: ignore[arg-type]
            )
            == NodeName.FALLBACK.value
        )


class TestHappyPath:
    async def test_checked_answer_is_explained(self, graph_factory: Any) -> None:
        state = await run_graph(graph_factory(ScriptedLLMClient(happy_script())))

        assert state["status"] == AnswerStatus.VERIFIED.value
        assert state["fallback_reason"] is None
        assert state["final_answer"] == EXPLANATION
        assert state["retry_count"] == 0
        assert state["confidence"] == pytest.approx(0.92)
        assert nodes_visited(state) == [
            NodeName.UNDERSTAND.value,
            NodeName.SOLVE.value,
            NodeName.SELF_CHECK.value,
            NodeName.EXPLAIN.value,
        ]
        assert decisions(state) == [
            RoutingDecision.TO_SOLVE.value,
            RoutingDecision.TO_SELF_CHECK.value,
            RoutingDecision.TO_EXPLAIN.value,
            RoutingDecision.END.value,
        ]

    async def test_step_sequences_are_contiguous(self, graph_factory: Any) -> None:
        state = await run_graph(graph_factory(ScriptedLLMClient(happy_script())))
        assert [step["sequence"] for step in state["trace"]] == [1, 2, 3, 4]

    async def test_understanding_flows_into_the_solve_prompt(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient(
            {
                **happy_script(),
                "understand": [
                    build_understanding(restated="Find the fall time", topic="free fall")
                ],
            }
        )
        await run_graph(graph_factory(llm))

        solve_prompt = llm.prompts_for("solve")[0]
        assert "Find the fall time" in solve_prompt
        assert "free fall" in solve_prompt

    async def test_conversation_history_reaches_the_understand_prompt(
        self, graph_factory: Any
    ) -> None:
        llm = ScriptedLLMClient(happy_script())
        history = [
            HistoryTurn(role=MessageRole.USER, content="We are studying free fall."),
            HistoryTurn(role=MessageRole.ASSISTANT, content="Great, ask away."),
        ]
        await run_graph(graph_factory(llm), history=history)

        prompt = llm.prompts_for("understand")[0]
        assert "We are studying free fall." in prompt
        assert "Great, ask away." in prompt

    async def test_checked_answer_flows_into_the_explain_prompt(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient({**happy_script(), "solve": [build_solution(answer="t = 2.02 s")]})
        await run_graph(graph_factory(llm))
        assert "t = 2.02 s" in llm.prompts_for("explain")[0]


class TestNotAnswerable:
    async def test_unanswerable_question_skips_straight_to_fallback(
        self, graph_factory: Any
    ) -> None:
        llm = ScriptedLLMClient(
            {**happy_script(), "understand": [build_understanding(is_answerable=False)]}
        )
        state = await run_graph(graph_factory(llm))

        assert state["status"] == AnswerStatus.UNVERIFIED.value
        assert state["fallback_reason"] == FallbackReason.NOT_ANSWERABLE.value
        assert nodes_visited(state) == [NodeName.UNDERSTAND.value, NodeName.FALLBACK.value]
        assert llm.call_count("solve") == 0
        assert "could not fully verify" in state["final_answer"]
        assert "not able to produce a usable attempt" in state["final_answer"]


class TestSelfCheckRetryLoop:
    async def test_failed_check_retries_the_solve_then_succeeds(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [
                    build_solution(answer="t = 9 s, which is wrong"),
                    build_solution(answer="t = 2.0 s, corrected"),
                ],
                "self_check": [
                    build_self_check(passed=False, confidence=0.3, issues=["Arithmetic error."]),
                    build_self_check(passed=True, confidence=0.91),
                ],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(graph_factory(llm))

        assert state["status"] == AnswerStatus.VERIFIED.value
        assert state["retry_count"] == 1
        assert nodes_visited(state) == [
            NodeName.UNDERSTAND.value,
            NodeName.SOLVE.value,
            NodeName.SELF_CHECK.value,
            NodeName.SOLVE.value,
            NodeName.SELF_CHECK.value,
            NodeName.EXPLAIN.value,
        ]
        assert RoutingDecision.RETRY_SOLVE.value in decisions(state)

    async def test_retry_prompt_carries_the_markers_objections(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution(answer="first attempt"), build_solution()],
                "self_check": [
                    build_self_check(passed=False, issues=["Used the wrong formula."]),
                    build_self_check(passed=True, confidence=0.9),
                ],
                "explain": [EXPLANATION],
            }
        )
        await run_graph(graph_factory(llm))

        prompts = llm.prompts_for("solve")
        assert len(prompts) == 2
        assert "Used the wrong formula." in prompts[1]
        assert "first attempt" in prompts[1]
        assert "REJECTED" in prompts[1]

    async def test_falls_back_once_retries_are_exhausted(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution(answer="still not right")],
                "self_check": [build_self_check(passed=False, issues=["Still wrong."])],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(graph_factory(llm, overrides={"max_solve_retries": 2}))

        assert state["status"] == AnswerStatus.UNVERIFIED.value
        assert state["fallback_reason"] == FallbackReason.SELF_CHECK_FAILED.value
        assert state["retry_count"] == 2
        # 1 initial solve + 2 retries
        assert llm.call_count("solve") == 3
        assert llm.call_count("self_check") == 3
        assert llm.call_count("explain") == 0
        assert "Still wrong." in state["final_answer"]
        assert "still not right" in state["final_answer"]

    async def test_zero_retries_falls_back_on_first_failure(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution()],
                "self_check": [build_self_check(passed=False, issues=["Nope."])],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(graph_factory(llm, overrides={"max_solve_retries": 0}))

        assert state["fallback_reason"] == FallbackReason.SELF_CHECK_FAILED.value
        assert llm.call_count("solve") == 1

    async def test_confidence_floor_overrides_a_passing_marker(self, graph_factory: Any) -> None:
        """``passed: true`` with low confidence must still be rejected."""
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution()],
                "self_check": [build_self_check(passed=True, confidence=0.2)],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(
            graph_factory(llm, overrides={"max_solve_retries": 0, "self_check_min_confidence": 0.6})
        )

        assert state["status"] == AnswerStatus.UNVERIFIED.value
        assert any("below the required" in issue for issue in state["self_check"]["issues"])

    async def test_deterministic_guardrail_overrides_a_passing_marker(
        self, graph_factory: Any
    ) -> None:
        """A model that approves a placeholder answer must still be overruled."""
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution(answer="The answer is TODO compute this")],
                "self_check": [build_self_check(passed=True, confidence=0.99)],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(graph_factory(llm, overrides={"max_solve_retries": 0}))

        assert state["status"] == AnswerStatus.UNVERIFIED.value
        assert any("placeholder" in issue.lower() for issue in state["self_check"]["issues"])

    async def test_answer_with_no_working_is_rejected(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution(answer="It is 2.0 seconds.", steps=[])],
                "self_check": [build_self_check(passed=True, confidence=0.95)],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(graph_factory(llm, overrides={"max_solve_retries": 0}))

        assert state["status"] == AnswerStatus.UNVERIFIED.value
        assert any("No working" in issue for issue in state["self_check"]["issues"])


class TestRetryAndFallbackOnLLMFailure:
    async def test_transient_failure_is_retried_then_succeeds(
        self, graph_factory: Any, sleep_calls: list[float]
    ) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [
                    LLMTransientError("502 from provider"),
                    build_understanding(),
                ],
                "solve": [build_solution()],
                "self_check": [build_self_check(passed=True, confidence=0.9)],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(graph_factory(llm))

        assert state["status"] == AnswerStatus.VERIFIED.value
        assert llm.call_count("understand") == 2
        assert len(sleep_calls) == 1

    async def test_exhausted_retries_fall_back(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [LLMTransientError("provider is down")],
                "solve": [build_solution()],
                "self_check": [build_self_check()],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(graph_factory(llm, overrides={"llm_max_attempts": 3}))

        assert state["status"] == AnswerStatus.UNVERIFIED.value
        assert state["fallback_reason"] == FallbackReason.NODE_ERROR.value
        assert llm.call_count("understand") == 3
        assert state["trace"][0]["status"] == "error"
        assert "provider is down" in state["trace"][0]["error"]

    async def test_permanent_error_is_not_retried(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [LLMPermanentError("invalid api key")],
                "solve": [build_solution()],
                "self_check": [build_self_check()],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(graph_factory(llm))

        assert state["fallback_reason"] == FallbackReason.NODE_ERROR.value
        assert llm.call_count("understand") == 1

    async def test_unparseable_response_is_retried_then_falls_back(
        self, graph_factory: Any
    ) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": ["I think this is a physics question, honestly"],
                "solve": [build_solution()],
                "self_check": [build_self_check()],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(graph_factory(llm, overrides={"llm_max_attempts": 2}))

        assert state["fallback_reason"] == FallbackReason.NODE_ERROR.value
        assert llm.call_count("understand") == 2

    async def test_solve_failure_skips_the_self_check(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [LLMTransientError("model overloaded")],
                "self_check": [build_self_check()],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(graph_factory(llm, overrides={"llm_max_attempts": 1}))

        assert state["fallback_reason"] == FallbackReason.NODE_ERROR.value
        assert nodes_visited(state) == [
            NodeName.UNDERSTAND.value,
            NodeName.SOLVE.value,
            NodeName.FALLBACK.value,
        ]
        assert llm.call_count("self_check") == 0

    async def test_explain_failure_still_falls_back_after_a_passing_check(
        self, graph_factory: Any
    ) -> None:
        """A checked answer that cannot be explained must not be lost silently."""
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution()],
                "self_check": [build_self_check(passed=True, confidence=0.95)],
                "explain": [LLMTransientError("explain endpoint down")],
            }
        )
        state = await run_graph(graph_factory(llm, overrides={"llm_max_attempts": 1}))

        assert state["status"] == AnswerStatus.UNVERIFIED.value
        assert state["fallback_reason"] == FallbackReason.NODE_ERROR.value
        assert nodes_visited(state)[-1] == NodeName.FALLBACK.value
        # The solved answer survives into the partial response.
        assert "2.0 seconds" in state["final_answer"]

    async def test_empty_explanation_is_treated_as_a_failure(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution()],
                "self_check": [build_self_check(passed=True, confidence=0.95)],
                "explain": ["   "],
            }
        )
        state = await run_graph(graph_factory(llm, overrides={"llm_max_attempts": 2}))

        assert state["fallback_reason"] == FallbackReason.NODE_ERROR.value
        assert llm.call_count("explain") == 2

    async def test_empty_solve_answer_is_retried(self, graph_factory: Any) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution(answer="")],
                "self_check": [build_self_check()],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(graph_factory(llm, overrides={"llm_max_attempts": 2}))

        assert state["fallback_reason"] == FallbackReason.NODE_ERROR.value
        assert llm.call_count("solve") == 2


class TestStepLimitGuard:
    async def test_step_budget_stops_the_retry_loop(self, graph_factory: Any) -> None:
        """A high retry allowance must still be bounded by max_graph_steps."""
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution(answer="never satisfactory")],
                "self_check": [build_self_check(passed=False, issues=["never satisfied"])],
                "explain": [EXPLANATION],
            }
        )
        state = await run_graph(
            graph_factory(llm, overrides={"max_solve_retries": 5, "max_graph_steps": 6})
        )

        assert state["status"] == AnswerStatus.UNVERIFIED.value
        assert state["fallback_reason"] == FallbackReason.STEP_LIMIT_REACHED.value
        assert decisions(state)[-2] == RoutingDecision.TO_FALLBACK_STEP_LIMIT.value

    async def test_guard_does_not_fire_on_a_normal_run(self, graph_factory: Any) -> None:
        state = await run_graph(
            graph_factory(ScriptedLLMClient(happy_script()), overrides={"max_graph_steps": 4})
        )
        assert state["status"] == AnswerStatus.VERIFIED.value
