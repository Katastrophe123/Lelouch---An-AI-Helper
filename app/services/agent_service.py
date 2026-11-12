"""Orchestrates one agent-mode turn: run the Claude agent, persist the result.

Mirrors :class:`~app.services.tutor_service.TutorService` so the API can route
a conversation to either engine with the same response shapes and the same SSE
event contract (``step`` / ``token`` / ``done`` / ``error``).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any

from app.assistant.events import Emitter
from app.assistant.runner import AgentOutcome, AgentRequest, AgentRunner
from app.config import Settings
from app.db.models import Conversation, Message
from app.db.repository import ChunkHit, ConversationRepository
from app.db.session import Database
from app.knowledge.service import KnowledgeService
from app.logging_config import bind_log_context, get_logger, unbind_log_context
from app.models.enums import AnswerStatus, StreamEventType
from app.services.tutor_service import ConversationNotFoundError, TurnResult, TutorService

logger = get_logger(__name__)

NO_ANSWER = "I wasn't able to finish this request."


class AgentService:
    def __init__(
        self,
        *,
        settings: Settings,
        runner: AgentRunner,
        knowledge: KnowledgeService,
    ) -> None:
        self._settings = settings
        self.runner = runner
        self._knowledge = knowledge

    def resolve_model(self, requested: str | None) -> str:
        """Validate a requested model against ``AGENT_MODELS``."""
        if requested is None:
            return self._settings.agent_model
        if requested not in self._settings.agent_models:
            raise ValueError(
                f"Model {requested!r} is not enabled; choose one of "
                f"{', '.join(self._settings.agent_models)}"
            )
        return requested

    def _searcher(self, user_id: uuid.UUID) -> Any:
        async def search(query: str, top_k: int | None) -> list[ChunkHit]:
            return await self._knowledge.search(user_id=user_id, query=query, top_k=top_k)

        return search

    async def _run(
        self,
        *,
        conversation: Conversation,
        content: str,
        repository: ConversationRepository,
        emitter: Emitter | None,
    ) -> tuple[Message, Message]:
        history = await repository.recent_history(
            conversation.id, limit=self._settings.history_window
        )
        user_message = await repository.add_user_message(conversation=conversation, content=content)
        request = AgentRequest(
            question=content,
            model=conversation.agent_model or self._settings.agent_model,
            user_id=conversation.user_id,
            history=history,
            resume_session_id=conversation.agent_session_id,
        )
        run_id = str(uuid.uuid4())
        bind_log_context(run_id=run_id, conversation_id=str(conversation.id))
        started = perf_counter()
        try:
            outcome = await self.runner.run(
                request, search_documents=self._searcher(conversation.user_id), emitter=emitter
            )
        finally:
            unbind_log_context("run_id", "conversation_id")
        assistant_message = await self._persist(
            conversation=conversation,
            repository=repository,
            outcome=outcome,
            latency_ms=max(0, int((perf_counter() - started) * 1000)),
            question=content,
        )
        return user_message, assistant_message

    async def _persist(
        self,
        *,
        conversation: Conversation,
        repository: ConversationRepository,
        outcome: AgentOutcome,
        latency_ms: int,
        question: str,
    ) -> Message:
        failed = outcome.is_error or not outcome.text
        if failed:
            content = outcome.text or NO_ANSWER
            if outcome.error:
                content = f"{content}\n\n**Problem:** {outcome.error}"
        else:
            content = outcome.text or NO_ANSWER
        message = await repository.add_assistant_message(
            conversation=conversation,
            content=content,
            answer_status=AnswerStatus.FAILED if failed else AnswerStatus.COMPLETED,
            retry_count=0,
            latency_ms=latency_ms,
            error=outcome.error,
            trace=outcome.steps,
        )
        if outcome.session_id:
            conversation.agent_session_id = outcome.session_id
        if conversation.title == "New conversation":
            conversation.title = TutorService.derive_title(question)
        await repository.touch_conversation(conversation)
        await repository.refresh_messages(conversation)
        logger.info(
            "agent_turn_completed",
            failed=failed,
            steps=len(outcome.steps),
            num_turns=outcome.num_turns,
            cost_usd=outcome.cost_usd,
            latency_ms=latency_ms,
        )
        return message

    # --- public API (same shape as TutorService) --------------------------
    async def run_turn(
        self, *, conversation: Conversation, content: str, repository: ConversationRepository
    ) -> TurnResult:
        user_message, assistant_message = await self._run(
            conversation=conversation, content=content, repository=repository, emitter=None
        )
        return TurnResult(user_message=user_message, assistant_message=assistant_message)

    async def stream_turn(
        self,
        *,
        database: Database,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        content: str,
    ) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def emitter(event: dict[str, Any]) -> None:
            await queue.put(event)

        async def worker() -> None:
            try:
                async with database.session() as session:
                    repository = ConversationRepository(session)
                    conversation = await repository.get_conversation(
                        conversation_id, user_id=user_id
                    )
                    if conversation is None:
                        raise ConversationNotFoundError(conversation_id)
                    user_message, assistant_message = await self._run(
                        conversation=conversation,
                        content=content,
                        repository=repository,
                        emitter=emitter,
                    )
                    await queue.put(
                        {
                            "event": StreamEventType.DONE.value,
                            "conversation_id": str(conversation.id),
                            "user_message_id": str(user_message.id),
                            "assistant_message_id": str(assistant_message.id),
                            "answer_status": assistant_message.answer_status,
                            "fallback_reason": None,
                            "confidence": None,
                            "retry_count": 0,
                            "latency_ms": assistant_message.latency_ms,
                            "content": assistant_message.content,
                        }
                    )
            except ConversationNotFoundError as exc:
                await queue.put(
                    {"event": StreamEventType.ERROR.value, "code": "not_found", "message": str(exc)}
                )
            except Exception as exc:  # noqa: BLE001 - the stream must close cleanly
                logger.exception("agent_stream_failed", error=str(exc))
                await queue.put(
                    {
                        "event": StreamEventType.ERROR.value,
                        "code": "internal_error",
                        "message": "The agent run failed before it could finish.",
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
