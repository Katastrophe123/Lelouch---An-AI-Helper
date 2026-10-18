"""Typed LangGraph state for the tutoring agent.

``trace`` uses an additive reducer so each node appends its own audit record
instead of overwriting the list; every other key is last-write-wins.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.models.domain import HistoryTurn, StepRecord
from app.models.enums import AnswerStatus


class TutorState(TypedDict, total=False):
    """State threaded through every node of the graph."""

    # --- immutable inputs -------------------------------------------------
    run_id: str
    conversation_id: str
    question: str
    history: list[dict[str, str]]

    # --- control flow -----------------------------------------------------
    step_count: int
    retry_count: int
    last_decision: str | None
    node_error: str | None

    # --- node outputs -----------------------------------------------------
    understanding: dict[str, Any] | None
    solution: dict[str, Any] | None
    self_check: dict[str, Any] | None
    explanation: str | None

    # --- terminal ---------------------------------------------------------
    status: str
    fallback_reason: str | None
    final_answer: str | None
    confidence: float | None

    # --- audit (append-only) ---------------------------------------------
    trace: Annotated[list[dict[str, Any]], operator.add]


def initial_state(
    *,
    run_id: str,
    conversation_id: str,
    question: str,
    history: list[HistoryTurn] | None = None,
) -> TutorState:
    """Build the starting state for one tutoring turn."""
    return TutorState(
        run_id=run_id,
        conversation_id=conversation_id,
        question=question,
        history=[turn.to_dict() for turn in (history or [])],
        step_count=0,
        retry_count=0,
        last_decision=None,
        node_error=None,
        understanding=None,
        solution=None,
        self_check=None,
        explanation=None,
        status=AnswerStatus.PENDING.value,
        fallback_reason=None,
        final_answer=None,
        confidence=None,
        trace=[],
    )


def history_turns(state: TutorState) -> list[HistoryTurn]:
    """Rehydrate the conversation history from state."""
    return [HistoryTurn.from_dict(item) for item in state.get("history", [])]


def trace_records(state: TutorState) -> list[StepRecord]:
    """Decode the accumulated trace back into :class:`StepRecord` objects."""
    return [StepRecord.from_dict(item) for item in state.get("trace", [])]
