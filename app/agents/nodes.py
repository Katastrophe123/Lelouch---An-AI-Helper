"""Graph node implementations.

Design notes
------------
* Every node is total: it never raises. Failures are captured as ``node_error``
  plus an error :class:`StepRecord`, and the router sends the turn to
  ``fallback``. That gives the graph a single, auditable fallback path.
* Each node computes the *routing decision* it implies and stores it on its own
  step record, so routers are pure lookups of ``state["last_decision"]`` and
  every decision is persisted with the message.
* The max-step guard is checked at the top of each node, which is what bounds
  the ``solve -> self_check -> solve`` retry loop.
* An optional ``emitter`` lets the ``explain`` node push tokens out as they
  arrive; that is what the SSE endpoint consumes. With no emitter the graph
  behaves identically, just without streaming.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from app.agents import prompts
from app.agents.state import TutorState, history_turns
from app.config import Settings
from app.llm.base import LLMClient, LLMError, LLMInvalidResponseError
from app.llm.retry import SleepFn, call_with_retries
from app.logging_config import get_logger
from app.models.domain import StepRecord
from app.models.enums import (
    AnswerStatus,
    Difficulty,
    FallbackReason,
    NodeName,
    RoutingDecision,
    StepStatus,
    StreamEventType,
    Subject,
)
from app.models.schemas import SelfCheck, Solution, Understanding

logger = get_logger(__name__)

StreamEvent = dict[str, Any]
Emitter = Callable[[StreamEvent], Awaitable[None]]

FALLBACK_REASON_BY_DECISION: dict[str, FallbackReason] = {
    RoutingDecision.TO_FALLBACK_CHECK_FAILED.value: FallbackReason.SELF_CHECK_FAILED,
    RoutingDecision.TO_FALLBACK_NODE_ERROR.value: FallbackReason.NODE_ERROR,
    RoutingDecision.TO_FALLBACK_STEP_LIMIT.value: FallbackReason.STEP_LIMIT_REACHED,
    RoutingDecision.TO_FALLBACK_NOT_ANSWERABLE.value: FallbackReason.NOT_ANSWERABLE,
}

MIN_ANSWER_CHARS = 8
_REFUSAL_MARKERS = ("i don't know", "i do not know", "i cannot help", "unable to answer")


@dataclass(frozen=True, slots=True)
class NodeConfig:
    """Behavioural knobs for the nodes, derived from :class:`Settings`."""

    self_check_min_confidence: float = 0.6
    max_solve_retries: int = 2
    max_graph_steps: int = 14
    llm_max_attempts: int = 3
    llm_backoff_base_seconds: float = 0.5
    history_window: int = 8

    @classmethod
    def from_settings(cls, settings: Settings) -> NodeConfig:
        return cls(
            self_check_min_confidence=settings.self_check_min_confidence,
            max_solve_retries=settings.max_solve_retries,
            max_graph_steps=settings.max_graph_steps,
            llm_max_attempts=settings.llm_max_attempts,
            llm_backoff_base_seconds=settings.llm_backoff_base_seconds,
            history_window=settings.history_window,
        )


# --------------------------------------------------------------------------- #
# Response coercion
# --------------------------------------------------------------------------- #
def _as_unit_float(value: Any, *, field: str) -> float:
    """Accept 0-1 floats, ``"0.82"`` strings and 0-100 percentages."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMInvalidResponseError(f"{field} is not numeric: {value!r}") from exc
    if number > 1.0:
        number = number / 100.0 if number <= 100.0 else 1.0
    return max(0.0, min(1.0, number))


def _as_str_list(value: Any, *, limit: int = 20, max_chars: int = 400) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:max_chars] for item in value[:limit]]


def coerce_understanding(data: dict[str, Any]) -> Understanding:
    """Normalise then validate an ``understand`` payload."""
    subject = str(data.get("subject", "")).strip().lower().replace(" ", "_")
    difficulty = str(data.get("difficulty", "")).strip().lower().replace(" ", "_")
    payload = {
        # An unrecognised label is not worth a retry: "other" is always valid
        # and the downstream nodes do not branch on subject.
        "subject": subject if subject in set(Subject) else Subject.OTHER.value,
        "difficulty": (
            difficulty if difficulty in set(Difficulty) else Difficulty.INTERMEDIATE.value
        ),
        "restated_question": str(data.get("restated_question", "")).strip()[:1000],
        "topic": str(data.get("topic", "")).strip()[:120],
        "knowns": _as_str_list(data.get("knowns")),
        "unknowns": _as_str_list(data.get("unknowns")),
        "is_answerable": bool(data.get("is_answerable", True)),
    }
    if not payload["restated_question"]:
        raise LLMInvalidResponseError("understand returned no restated_question")
    try:
        return Understanding.model_validate(payload)
    except ValidationError as exc:
        raise LLMInvalidResponseError(f"Unusable understanding payload: {payload}") from exc


def coerce_solution(data: dict[str, Any]) -> Solution:
    """Normalise then validate a ``solve`` payload."""
    payload = {
        "answer": str(data.get("answer", "")).strip()[:4000],
        "steps": _as_str_list(data.get("steps"), limit=30, max_chars=1000),
        "assumptions": _as_str_list(data.get("assumptions")),
    }
    if not payload["answer"]:
        raise LLMInvalidResponseError("solve returned an empty answer")
    try:
        return Solution.model_validate(payload)
    except ValidationError as exc:
        raise LLMInvalidResponseError(f"Unusable solution payload: {payload}") from exc


def coerce_self_check(data: dict[str, Any]) -> SelfCheck:
    """Normalise then validate a ``self_check`` payload."""
    payload = {
        "passed": bool(data.get("passed", False)),
        "confidence": _as_unit_float(data.get("confidence", 0.0), field="confidence"),
        "issues": _as_str_list(data.get("issues")),
    }
    try:
        return SelfCheck.model_validate(payload)
    except ValidationError as exc:
        raise LLMInvalidResponseError(f"Unusable self-check payload: {payload}") from exc


def deterministic_answer_issues(solution: Solution) -> list[str]:
    """Checks that do not depend on the model's own judgement."""
    issues: list[str] = []
    answer = solution.answer.strip()
    if len(answer) < MIN_ANSWER_CHARS:
        issues.append("The answer is too short to be a real answer.")
    lowered = answer.lower()
    if any(marker in lowered for marker in _REFUSAL_MARKERS):
        issues.append("The answer declines to answer the question.")
    if "TODO" in answer or "[insert" in lowered:
        issues.append("The answer contains an unresolved placeholder.")
    if not solution.steps:
        issues.append("No working was provided, so the answer cannot be explained.")
    return issues


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
class TutorNodes:
    """The callables wired together by :func:`app.agents.graph.build_graph`."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        config: NodeConfig,
        sleep: SleepFn | None = None,
        emitter: Emitter | None = None,
    ) -> None:
        self._llm = llm
        self.config = config
        self._sleep = sleep
        self._emitter = emitter

    # --- shared plumbing --------------------------------------------------
    async def _retry(self, operation: Any, name: str) -> Any:
        return await call_with_retries(
            operation,
            max_attempts=self.config.llm_max_attempts,
            base_delay=self.config.llm_backoff_base_seconds,
            operation_name=name,
            sleep=self._sleep,
        )

    async def _emit(self, event: StreamEvent) -> None:
        if self._emitter is not None:
            await self._emitter(event)

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int((perf_counter() - started) * 1000))

    def _step_limit_hit(
        self, *, sequence: int, node: NodeName, started: float
    ) -> dict[str, Any] | None:
        """Return a fallback update when the step budget is exhausted."""
        if sequence <= self.config.max_graph_steps:
            return None
        return self._failed(
            sequence=sequence,
            node=node,
            started=started,
            error=f"Max graph steps ({self.config.max_graph_steps}) exceeded",
            decision=RoutingDecision.TO_FALLBACK_STEP_LIMIT,
        )

    def _ok(
        self,
        *,
        sequence: int,
        node: NodeName,
        started: float,
        output: dict[str, Any],
        decision: RoutingDecision,
    ) -> dict[str, Any]:
        record = StepRecord(
            sequence=sequence,
            node=node,
            status=StepStatus.OK,
            latency_ms=self._elapsed_ms(started),
            output=output,
            routing_decision=decision,
        )
        logger.info(
            "graph_node_completed",
            node=node.value,
            sequence=sequence,
            decision=decision.value,
            latency_ms=record.latency_ms,
        )
        return {
            "step_count": sequence,
            "last_decision": decision.value,
            "node_error": None,
            "trace": [record.to_dict()],
            "_event": record,
        }

    def _failed(
        self,
        *,
        sequence: int,
        node: NodeName,
        started: float,
        error: str,
        decision: RoutingDecision,
    ) -> dict[str, Any]:
        record = StepRecord(
            sequence=sequence,
            node=node,
            status=StepStatus.ERROR,
            latency_ms=self._elapsed_ms(started),
            routing_decision=decision,
            error=error[:2000],
        )
        logger.warning(
            "graph_node_failed",
            node=node.value,
            sequence=sequence,
            decision=decision.value,
            error=error[:500],
        )
        return {
            "step_count": sequence,
            "last_decision": decision.value,
            "node_error": error[:2000],
            "trace": [record.to_dict()],
            "_event": record,
        }

    async def _finish(self, update: dict[str, Any]) -> dict[str, Any]:
        """Emit a step event and strip the internal ``_event`` key."""
        record: StepRecord | None = update.pop("_event", None)
        if record is not None:
            await self._emit(
                {
                    "event": StreamEventType.STEP.value,
                    "node": record.node.value,
                    "sequence": record.sequence,
                    "status": record.status.value,
                    "decision": (
                        record.routing_decision.value if record.routing_decision else None
                    ),
                    "latency_ms": record.latency_ms,
                }
            )
        return update

    # --- 1. understand ----------------------------------------------------
    async def understand(self, state: TutorState) -> dict[str, Any]:
        sequence = state.get("step_count", 0) + 1
        started = perf_counter()
        limited = self._step_limit_hit(sequence=sequence, node=NodeName.UNDERSTAND, started=started)
        if limited:
            return await self._finish(limited)

        question = state.get("question", "")
        history = history_turns(state)

        async def operation() -> Understanding:
            data = await self._llm.generate_json(
                system=prompts.UNDERSTAND_SYSTEM,
                user=prompts.build_understand_user(question=question, history=history),
                temperature=0.0,
                max_tokens=500,
            )
            return coerce_understanding(data)

        try:
            result = await self._retry(operation, "understand")
        except LLMError as exc:
            return await self._finish(
                self._failed(
                    sequence=sequence,
                    node=NodeName.UNDERSTAND,
                    started=started,
                    error=f"Understanding the question failed: {exc}",
                    decision=RoutingDecision.TO_FALLBACK_NODE_ERROR,
                )
            )

        decision = (
            RoutingDecision.TO_SOLVE
            if result.is_answerable
            else RoutingDecision.TO_FALLBACK_NOT_ANSWERABLE
        )
        understanding = result.model_dump(mode="json")
        update = {
            "understanding": understanding,
            **self._ok(
                sequence=sequence,
                node=NodeName.UNDERSTAND,
                started=started,
                output=understanding,
                decision=decision,
            ),
        }
        return await self._finish(update)

    # --- 2. solve ---------------------------------------------------------
    async def solve(self, state: TutorState) -> dict[str, Any]:
        sequence = state.get("step_count", 0) + 1
        started = perf_counter()
        limited = self._step_limit_hit(sequence=sequence, node=NodeName.SOLVE, started=started)
        if limited:
            return await self._finish(limited)

        question = state.get("question", "")
        understanding = state.get("understanding") or {}
        previous_solution = state.get("solution") or {}
        previous_check = state.get("self_check") or {}

        user_prompt = prompts.build_solve_user(
            question=question,
            understanding=understanding,
            previous_answer=previous_solution.get("answer"),
            previous_issues=list(previous_check.get("issues") or []),
        )

        async def operation() -> Solution:
            data = await self._llm.generate_json(
                system=prompts.SOLVE_SYSTEM,
                user=user_prompt,
                temperature=0.1,
                max_tokens=1200,
            )
            return coerce_solution(data)

        try:
            result = await self._retry(operation, "solve")
        except LLMError as exc:
            return await self._finish(
                self._failed(
                    sequence=sequence,
                    node=NodeName.SOLVE,
                    started=started,
                    error=f"Solving failed: {exc}",
                    decision=RoutingDecision.TO_FALLBACK_NODE_ERROR,
                )
            )

        solution = result.model_dump(mode="json")
        update = {
            "solution": solution,
            **self._ok(
                sequence=sequence,
                node=NodeName.SOLVE,
                started=started,
                output={
                    "answer_preview": result.answer[:280],
                    "step_count": len(result.steps),
                    "attempt": state.get("retry_count", 0) + 1,
                },
                decision=RoutingDecision.TO_SELF_CHECK,
            ),
        }
        return await self._finish(update)

    # --- 3. self-check ----------------------------------------------------
    async def self_check(self, state: TutorState) -> dict[str, Any]:
        sequence = state.get("step_count", 0) + 1
        started = perf_counter()
        retry_count = state.get("retry_count", 0)
        limited = self._step_limit_hit(sequence=sequence, node=NodeName.SELF_CHECK, started=started)
        if limited:
            return await self._finish(limited)

        question = state.get("question", "")
        raw_solution = state.get("solution") or {}
        solution = Solution.model_validate(raw_solution)

        async def operation() -> SelfCheck:
            data = await self._llm.generate_json(
                system=prompts.SELF_CHECK_SYSTEM,
                user=prompts.build_self_check_user(question=question, solution=raw_solution),
                temperature=0.0,
                max_tokens=500,
            )
            return coerce_self_check(data)

        try:
            result = await self._retry(operation, "self_check")
        except LLMError as exc:
            return await self._finish(
                self._failed(
                    sequence=sequence,
                    node=NodeName.SELF_CHECK,
                    started=started,
                    error=f"Self-check failed: {exc}",
                    decision=RoutingDecision.TO_FALLBACK_NODE_ERROR,
                )
            )

        # The model's verdict is ANDed with hard checks and a confidence floor,
        # so an agreeable marker cannot wave through a bad answer.
        issues = list(result.issues) + deterministic_answer_issues(solution)
        below_floor = result.confidence < self.config.self_check_min_confidence
        if below_floor:
            issues.append(
                f"Confidence {result.confidence:.2f} is below the required "
                f"{self.config.self_check_min_confidence:.2f}."
            )
        passed = result.passed and not issues

        check = SelfCheck(passed=passed, confidence=result.confidence, issues=issues).model_dump(
            mode="json"
        )

        if passed:
            decision = RoutingDecision.TO_EXPLAIN
            extra: dict[str, Any] = {}
        elif retry_count < self.config.max_solve_retries:
            decision = RoutingDecision.RETRY_SOLVE
            extra = {"retry_count": retry_count + 1}
        else:
            decision = RoutingDecision.TO_FALLBACK_CHECK_FAILED
            extra = {}

        update = {
            "self_check": check,
            "confidence": result.confidence,
            **extra,
            **self._ok(
                sequence=sequence,
                node=NodeName.SELF_CHECK,
                started=started,
                output={
                    "passed": passed,
                    "confidence": result.confidence,
                    "issues": issues,
                    "retry_count": retry_count,
                },
                decision=decision,
            ),
        }
        return await self._finish(update)

    # --- 4a. explain ------------------------------------------------------
    async def explain(self, state: TutorState) -> dict[str, Any]:
        sequence = state.get("step_count", 0) + 1
        started = perf_counter()
        limited = self._step_limit_hit(sequence=sequence, node=NodeName.EXPLAIN, started=started)
        if limited:
            return await self._finish(limited)

        question = state.get("question", "")
        understanding = state.get("understanding") or {}
        solution = state.get("solution") or {}
        user_prompt = prompts.build_explain_user(
            question=question, understanding=understanding, solution=solution
        )

        async def operation() -> str:
            if self._emitter is None:
                response = await self._llm.generate(
                    system=prompts.EXPLAIN_SYSTEM,
                    user=user_prompt,
                    temperature=0.3,
                    max_tokens=1200,
                )
                text = response.text.strip()
            else:
                # Stream so the client sees the explanation being written.
                chunks: list[str] = []
                async for piece in self._llm.stream(
                    system=prompts.EXPLAIN_SYSTEM,
                    user=user_prompt,
                    temperature=0.3,
                    max_tokens=1200,
                ):
                    chunks.append(piece)
                    await self._emit({"event": StreamEventType.TOKEN.value, "text": piece})
                text = "".join(chunks).strip()
            if not text:
                raise LLMInvalidResponseError("Explanation was empty")
            return text

        try:
            explanation = await self._retry(operation, "explain")
        except LLMError as exc:
            return await self._finish(
                self._failed(
                    sequence=sequence,
                    node=NodeName.EXPLAIN,
                    started=started,
                    error=f"Explaining failed: {exc}",
                    decision=RoutingDecision.TO_FALLBACK_NODE_ERROR,
                )
            )

        update = {
            "explanation": explanation,
            "final_answer": explanation,
            "status": AnswerStatus.VERIFIED.value,
            "fallback_reason": None,
            **self._ok(
                sequence=sequence,
                node=NodeName.EXPLAIN,
                started=started,
                output={"explanation_chars": len(explanation)},
                decision=RoutingDecision.END,
            ),
        }
        return await self._finish(update)

    # --- 4b. fallback -----------------------------------------------------
    async def fallback(self, state: TutorState) -> dict[str, Any]:
        sequence = state.get("step_count", 0) + 1
        started = perf_counter()
        decision_value = state.get("last_decision") or ""
        reason = FALLBACK_REASON_BY_DECISION.get(decision_value, FallbackReason.NODE_ERROR)

        check = state.get("self_check") or {}
        issues = list(check.get("issues") or [])
        node_error = state.get("node_error")
        if node_error:
            issues.append(node_error)

        answer = prompts.build_fallback_answer(
            reason=reason.value,
            question=state.get("question", ""),
            solution=state.get("solution"),
            issues=issues,
        )

        update = {
            "status": AnswerStatus.UNVERIFIED.value,
            "fallback_reason": reason.value,
            "final_answer": answer,
            **self._ok(
                sequence=sequence,
                node=NodeName.FALLBACK,
                started=started,
                output={"reason": reason.value, "issue_count": len(issues)},
                decision=RoutingDecision.END,
            ),
            # Keep the originating error on the state (``_ok`` clears it) so the
            # service can persist it on the message row, not just the trace.
            "node_error": node_error,
        }
        return await self._finish(update)
