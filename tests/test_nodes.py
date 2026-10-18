"""Unit tests for node internals the end-to-end routing tests cannot pin down."""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.nodes import (
    NodeConfig,
    TutorNodes,
    coerce_self_check,
    coerce_solution,
    coerce_understanding,
    deterministic_answer_issues,
)
from app.agents.state import history_turns, initial_state, trace_records
from app.config import Settings
from app.llm.base import LLMInvalidResponseError
from app.models.domain import HistoryTurn
from app.models.enums import (
    Difficulty,
    MessageRole,
    NodeName,
    RoutingDecision,
    StepStatus,
    Subject,
)
from app.models.schemas import Solution
from tests.fakes import ScriptedLLMClient, happy_script


class TestCoerceUnderstanding:
    def test_normalises_case_and_spacing(self) -> None:
        result = coerce_understanding(
            {
                "restated_question": "Find t",
                "subject": " Computer Science ",
                "difficulty": "University",
            }
        )
        assert result.subject is Subject.COMPUTER_SCIENCE
        assert result.difficulty is Difficulty.UNIVERSITY

    def test_unknown_subject_degrades_to_other(self) -> None:
        """An unrecognised label is not worth a retry; nothing branches on it."""
        result = coerce_understanding({"restated_question": "Find t", "subject": "astrophysics"})
        assert result.subject is Subject.OTHER

    def test_unknown_difficulty_degrades_to_the_default(self) -> None:
        result = coerce_understanding(
            {"restated_question": "Find t", "subject": "math", "difficulty": "impossible"}
        )
        assert result.difficulty is Difficulty.INTERMEDIATE

    def test_missing_restated_question_is_invalid(self) -> None:
        with pytest.raises(LLMInvalidResponseError, match="restated_question"):
            coerce_understanding({"subject": "math"})

    def test_is_answerable_defaults_to_true(self) -> None:
        result = coerce_understanding({"restated_question": "Find t", "subject": "math"})
        assert result.is_answerable is True

    def test_non_list_knowns_are_discarded(self) -> None:
        result = coerce_understanding(
            {"restated_question": "Find t", "subject": "math", "knowns": "20 m"}
        )
        assert result.knowns == []

    def test_long_fields_are_truncated(self) -> None:
        result = coerce_understanding(
            {
                "restated_question": "x" * 2000,
                "subject": "math",
                "topic": "y" * 500,
            }
        )
        assert len(result.restated_question) == 1000
        assert len(result.topic) == 120


class TestCoerceSolution:
    def test_empty_answer_is_invalid(self) -> None:
        with pytest.raises(LLMInvalidResponseError, match="empty answer"):
            coerce_solution({"answer": "   ", "steps": ["a"]})

    def test_steps_are_capped_and_truncated(self) -> None:
        result = coerce_solution(
            {"answer": "42", "steps": ["s" * 2000] + [f"step {i}" for i in range(40)]}
        )
        assert len(result.steps) == 30
        assert len(result.steps[0]) == 1000

    def test_non_list_steps_become_empty(self) -> None:
        result = coerce_solution({"answer": "42", "steps": "just do it"})
        assert result.steps == []


class TestCoerceSelfCheck:
    def test_missing_fields_default_to_failure(self) -> None:
        result = coerce_self_check({})
        assert result.passed is False
        assert result.confidence == 0.0

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(0.75, 0.75), ("0.42", 0.42), (85, 0.85), (100, 1.0), (150, 1.0), (-1, 0.0)],
    )
    def test_confidence_coercion(self, raw: Any, expected: float) -> None:
        result = coerce_self_check({"passed": True, "confidence": raw})
        assert result.confidence == pytest.approx(expected)

    def test_rejects_non_numeric_confidence(self) -> None:
        with pytest.raises(LLMInvalidResponseError, match="confidence"):
            coerce_self_check({"passed": True, "confidence": "very sure"})

    def test_issues_are_stringified(self) -> None:
        result = coerce_self_check({"passed": False, "confidence": 0.1, "issues": [1, None]})
        assert result.issues == ["1", "None"]


class TestDeterministicAnswerIssues:
    def test_good_answer_has_no_issues(self) -> None:
        solution = Solution(answer="The time taken is 2.0 seconds.", steps=["Use s=ut+½at²."])
        assert deterministic_answer_issues(solution) == []

    @pytest.mark.parametrize(
        ("answer", "fragment"),
        [
            ("42", "too short"),
            ("I don't know how to work this one out at all", "declines"),
            ("The result is TODO finish this calculation", "placeholder"),
            ("The value is [insert number here] metres per second", "placeholder"),
        ],
    )
    def test_bad_answers_are_flagged(self, answer: str, fragment: str) -> None:
        solution = Solution(answer=answer, steps=["a step"])
        issues = deterministic_answer_issues(solution)
        assert any(fragment in issue.lower() for issue in issues)

    def test_missing_working_is_flagged(self) -> None:
        solution = Solution(answer="The time taken is 2.0 seconds.", steps=[])
        issues = deterministic_answer_issues(solution)
        assert any("No working" in issue for issue in issues)


class TestStateHelpers:
    def test_initial_state_has_all_control_keys(self) -> None:
        state = initial_state(run_id="r", conversation_id="c", question="q")
        assert state["step_count"] == 0
        assert state["retry_count"] == 0
        assert state["status"] == "pending"
        assert state["trace"] == []
        assert state["history"] == []

    def test_history_roundtrips_through_state(self) -> None:
        history = [
            HistoryTurn(role=MessageRole.USER, content="hello"),
            HistoryTurn(role=MessageRole.ASSISTANT, content="hi"),
        ]
        state = initial_state(run_id="r", conversation_id="c", question="q", history=history)
        assert history_turns(state) == history

    def test_history_from_empty_state_is_empty(self) -> None:
        assert history_turns({}) == []


class TestNodeBehaviour:
    @pytest.fixture
    def nodes(self, settings: Settings, fake_sleep: Any) -> TutorNodes:
        return TutorNodes(
            llm=ScriptedLLMClient(happy_script()),
            config=NodeConfig.from_settings(settings),
            sleep=fake_sleep,
        )

    def test_config_is_derived_from_settings(self, settings: Settings) -> None:
        config = NodeConfig.from_settings(settings)
        assert config.self_check_min_confidence == settings.self_check_min_confidence
        assert config.max_solve_retries == settings.max_solve_retries
        assert config.max_graph_steps == settings.max_graph_steps
        assert config.history_window == settings.history_window

    async def test_understand_records_a_step_with_latency(self, nodes: TutorNodes) -> None:
        state = initial_state(run_id="r", conversation_id="c", question="What is 2 + 2?")
        update = await nodes.understand(state)

        assert update["step_count"] == 1
        assert update["last_decision"] == RoutingDecision.TO_SOLVE.value
        [record] = trace_records({"trace": update["trace"]})
        assert record.node is NodeName.UNDERSTAND
        assert record.status is StepStatus.OK
        assert record.latency_ms >= 0

    async def test_internal_event_key_is_not_leaked_into_state(self, nodes: TutorNodes) -> None:
        """``_event`` is plumbing for the emitter and must not reach the graph."""
        state = initial_state(run_id="r", conversation_id="c", question="What is 2 + 2?")
        update = await nodes.understand(state)
        assert "_event" not in update

    async def test_step_limit_guard_short_circuits_before_the_llm(self, nodes: TutorNodes) -> None:
        state = initial_state(run_id="r", conversation_id="c", question="q")
        state["step_count"] = nodes.config.max_graph_steps

        update = await nodes.understand(state)

        assert update["last_decision"] == RoutingDecision.TO_FALLBACK_STEP_LIMIT.value
        assert update["node_error"] is not None
        [record] = trace_records({"trace": update["trace"]})
        assert record.status is StepStatus.ERROR

    async def test_fallback_preserves_the_node_error_for_persistence(
        self, nodes: TutorNodes
    ) -> None:
        state = initial_state(run_id="r", conversation_id="c", question="q")
        state["node_error"] = "Solving failed: provider down"
        state["last_decision"] = RoutingDecision.TO_FALLBACK_NODE_ERROR.value

        update = await nodes.fallback(state)

        assert update["node_error"] == "Solving failed: provider down"
        assert update["fallback_reason"] == "node_error"
        assert "provider down" in update["final_answer"]

    async def test_fallback_without_a_solution_says_so(self, nodes: TutorNodes) -> None:
        state = initial_state(run_id="r", conversation_id="c", question="q")
        state["last_decision"] = RoutingDecision.TO_FALLBACK_NOT_ANSWERABLE.value

        update = await nodes.fallback(state)

        assert "not able to produce a usable attempt" in update["final_answer"]

    async def test_emitter_receives_one_step_event_per_node(
        self, settings: Settings, fake_sleep: Any
    ) -> None:
        events: list[dict[str, Any]] = []

        async def emitter(event: dict[str, Any]) -> None:
            events.append(event)

        nodes = TutorNodes(
            llm=ScriptedLLMClient(happy_script()),
            config=NodeConfig.from_settings(settings),
            sleep=fake_sleep,
            emitter=emitter,
        )
        state = initial_state(run_id="r", conversation_id="c", question="What is 2 + 2?")
        await nodes.understand(state)

        assert [event["event"] for event in events] == ["step"]
        assert events[0]["node"] == "understand"
        assert events[0]["decision"] == RoutingDecision.TO_SOLVE.value
