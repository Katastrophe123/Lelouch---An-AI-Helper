"""Orchestrates one tutoring turn: graph execution plus durable persistence."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langgraph.errors import GraphRecursionError

from app.agents.graph import build_tutor_graph
from app.agents.state import TutorState, initial_state, trace_records
from app.config import Settings
from app.db.models import Conversation, Message
from app.db.repository import ConversationRepository
from app.db.session import Database
from app.llm.base import LLMClient
from app.llm.retry import SleepFn
from app.logging_config import bind_log_context, get_logger, unbind_log_context
from app.models.enums import (
    AnswerStatus,
    FallbackReason,
    StreamEventType,
    Subject,
)

logger = get_logger(__name__)

MAX_TITLE_CHARS = 60


@dataclass(slots=True)
class TurnResult:
    """Both messages produced by one tutoring turn."""

    user_message: Message
    assistant_message: Message


class ConversationNotFoundError(Exception):
    """Raised when a conversation does not exist or is not owned by the caller."""

    def __init__(self, conversation_id: uuid.UUID) -> None:
        super().__init__(f"Conversation {conversation_id} was not found")
        self.conversation_id = conversation_id


class TutorService:
    """Runs the tutoring graph and persists the resulting messages.

    The non-streaming graph is built once per process (it is stateless); the
    streaming path builds a fresh graph per request so the emitter can be bound
    to that request's event queue.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMClient,
        graph: Any | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._sleep = sleep
        self._graph = graph or build_tutor_graph(settings=settings, llm=llm, sleep=sleep)

    @property
    def recursion_limit(self) -> int:
        """LangGraph's own loop guard, kept above our in-node step budget.

        The in-node ``max_graph_steps`` check is the primary guard because it
        produces an auditable fallback; this is the backstop.
        """
        return self._settings.max_graph_steps + 6

    # --- non-streaming ----------------------------------------------------
    async def run_turn(
        self,
        *,
        conversation: Conversation,
        content: str,
        repository: ConversationRepository,
    ) -> TurnResult:
        """Store the question, run the graph, store the answer."""
        # History is read *before* the new question is stored, so the prompt
        # never contains the question twice.
        history = await repository.recent_history(
            conversation.id, limit=self._settings.history_window
        )
        user_message = await repository.add_user_message(conversation=conversation, content=content)

        run_id = str(uuid.uuid4())
        bind_log_context(run_id=run_id, conversation_id=str(conversation.id))
        started = perf_counter()
        try:
            final_state = await self._invoke(
                graph=self._graph,
                run_id=run_id,
                conversation_id=str(conversation.id),
                question=content,
                history=history,
            )
        finally:
            unbind_log_context("run_id", "conversation_id")

        assistant_message = await self._persist_turn(
            conversation=conversation,
            repository=repository,
            final_state=final_state,
            latency_ms=self._elapsed_ms(started),
            question=content,
        )
        return TurnResult(user_message=user_message, assistant_message=assistant_message)

    # --- streaming --------------------------------------------------------
    async def stream_turn(
        self,
        *,
        database: Database,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        content: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE payloads while the graph runs.

        The work happens in a background task that pushes onto a queue, which
        this generator drains. The task owns its own database session rather
        than the request-scoped one, because a streaming response outlives the
        request-scoped dependency teardown.
        """
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def emitter(event: dict[str, Any]) -> None:
            await queue.put(event)

        graph = build_tutor_graph(
            settings=self._settings, llm=self._llm, sleep=self._sleep, emitter=emitter
        )
        run_id = str(uuid.uuid4())

        async def worker() -> None:
            try:
                async with database.session() as session:
                    repository = ConversationRepository(session)
                    conversation = await repository.get_conversation(
                        conversation_id, user_id=user_id
                    )
                    if conversation is None:
                        raise ConversationNotFoundError(conversation_id)

                    history = await repository.recent_history(
                        conversation.id, limit=self._settings.history_window
                    )
                    user_message = await repository.add_user_message(
                        conversation=conversation, content=content
                    )

                    started = perf_counter()
                    final_state = await self._invoke(
                        graph=graph,
                        run_id=run_id,
                        conversation_id=str(conversation.id),
                        question=content,
                        history=history,
                    )
                    assistant_message = await self._persist_turn(
                        conversation=conversation,
                        repository=repository,
                        final_state=final_state,
                        latency_ms=self._elapsed_ms(started),
                        question=content,
                    )
                    await queue.put(
                        {
                            "event": StreamEventType.DONE.value,
                            "conversation_id": str(conversation.id),
                            "user_message_id": str(user_message.id),
                            "assistant_message_id": str(assistant_message.id),
                            "answer_status": assistant_message.answer_status,
                            "fallback_reason": assistant_message.fallback_reason,
                            "confidence": assistant_message.confidence,
                            "retry_count": assistant_message.retry_count,
                            "latency_ms": assistant_message.latency_ms,
                            "content": assistant_message.content,
                        }
                    )
            except ConversationNotFoundError as exc:
                await queue.put(
                    {
                        "event": StreamEventType.ERROR.value,
                        "code": "not_found",
                        "message": str(exc),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - the stream must close cleanly
                logger.exception("stream_turn_failed", error=str(exc))
                await queue.put(
                    {
                        "event": StreamEventType.ERROR.value,
                        "code": "internal_error",
                        "message": "The tutoring run failed before it could finish.",
                    }
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(worker())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # --- shared -----------------------------------------------------------
    async def _invoke(
        self,
        *,
        graph: Any,
        run_id: str,
        conversation_id: str,
        question: str,
        history: list[Any],
    ) -> TutorState:
        """Run the graph, converting its own guard errors into a fallback state."""
        state = initial_state(
            run_id=run_id,
            conversation_id=conversation_id,
            question=question,
            history=history,
        )
        try:
            result: TutorState = await graph.ainvoke(
                state, config={"recursion_limit": self.recursion_limit}
            )
        except GraphRecursionError as exc:
            logger.error("graph_recursion_limit_reached", error=str(exc))
            return TutorState(
                **{
                    **state,
                    "status": AnswerStatus.UNVERIFIED.value,
                    "fallback_reason": FallbackReason.STEP_LIMIT_REACHED.value,
                    "final_answer": (
                        "**I could not finish checking this answer, so please verify it "
                        "yourself.** The tutoring run exceeded its step budget."
                    ),
                    "node_error": f"Graph recursion limit reached: {exc}",
                }
            )
        return result

    async def _persist_turn(
        self,
        *,
        conversation: Conversation,
        repository: ConversationRepository,
        final_state: TutorState,
        latency_ms: int,
        question: str,
    ) -> Message:
        """Write the assistant message, its trace, and refresh the conversation."""
        records = trace_records(final_state)
        understanding = final_state.get("understanding") or {}
        raw_subject = understanding.get("subject")
        subject = Subject(raw_subject) if raw_subject in set(Subject) else None

        status = AnswerStatus(final_state.get("status", AnswerStatus.FAILED.value))
        reason = final_state.get("fallback_reason")
        content = final_state.get("final_answer") or (
            "I was unable to produce an answer for this question."
        )

        assistant_message = await repository.add_assistant_message(
            conversation=conversation,
            content=content,
            answer_status=status,
            fallback_reason=FallbackReason(reason) if reason else None,
            confidence=final_state.get("confidence"),
            retry_count=int(final_state.get("retry_count", 0)),
            latency_ms=latency_ms,
            error=final_state.get("node_error"),
            trace=records,
        )

        # First real question becomes the conversation title, so the sidebar is
        # readable without the user having to name every thread.
        if conversation.title == "New conversation":
            conversation.title = self.derive_title(question)
        await repository.touch_conversation(conversation, subject=subject)
        await repository.refresh_messages(conversation)

        logger.info(
            "tutor_turn_completed",
            status=status.value,
            fallback_reason=reason,
            confidence=final_state.get("confidence"),
            retry_count=int(final_state.get("retry_count", 0)),
            steps=len(records),
            latency_ms=latency_ms,
        )
        return assistant_message

    @staticmethod
    def derive_title(question: str) -> str:
        """Condense a question into a short conversation title."""
        collapsed = " ".join(question.split())
        if len(collapsed) <= MAX_TITLE_CHARS:
            return collapsed or "New conversation"
        return collapsed[: MAX_TITLE_CHARS - 1].rstrip() + "…"

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int((perf_counter() - started) * 1000))
