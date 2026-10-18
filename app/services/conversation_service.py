"""Read-side service for conversations and messages."""

from __future__ import annotations

import math
import uuid

from app.db.models import Conversation, Message
from app.db.repository import ConversationRepository
from app.models.domain import StepRecord
from app.models.enums import ConversationStatus, Subject
from app.models.schemas import (
    ConversationDetail,
    ConversationSummary,
    MessageResponse,
    Page,
    StepResponse,
)
from app.services.tutor_service import ConversationNotFoundError


def message_to_response(message: Message) -> MessageResponse:
    """Map a message row onto its API shape, decoding the stored trace."""
    trace = [
        StepResponse.model_validate(StepRecord.from_dict(item).to_dict())
        for item in (message.trace or [])
    ]
    return MessageResponse(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,  # type: ignore[arg-type]
        content=message.content,
        created_at=message.created_at,
        answer_status=message.answer_status,  # type: ignore[arg-type]
        fallback_reason=message.fallback_reason,  # type: ignore[arg-type]
        confidence=message.confidence,
        retry_count=message.retry_count,
        latency_ms=message.latency_ms,
        error=message.error,
        trace=trace,
    )


def conversation_to_detail(conversation: Conversation) -> ConversationDetail:
    return ConversationDetail(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        subject=conversation.subject,  # type: ignore[arg-type]
        status=conversation.status,  # type: ignore[arg-type]
        message_count=conversation.message_count,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[message_to_response(message) for message in conversation.messages],
    )


class ConversationService:
    """Maps persisted conversations onto API response models."""

    def __init__(self, repository: ConversationRepository) -> None:
        self._repository = repository

    async def create(
        self,
        *,
        user_id: uuid.UUID,
        title: str | None = None,
        subject: Subject | None = None,
    ) -> ConversationDetail:
        conversation = await self._repository.create_conversation(
            user_id=user_id, title=title, subject=subject
        )
        await self._repository.refresh_messages(conversation)
        return conversation_to_detail(conversation)

    async def get(self, conversation_id: uuid.UUID, *, user_id: uuid.UUID) -> ConversationDetail:
        conversation = await self._repository.get_conversation(conversation_id, user_id=user_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation_to_detail(conversation)

    async def require_conversation(
        self, conversation_id: uuid.UUID, *, user_id: uuid.UUID
    ) -> Conversation:
        """Return the ORM row, for callers that need to write to it."""
        conversation = await self._repository.get_conversation(conversation_id, user_id=user_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    async def list_for_user(
        self,
        *,
        user_id: uuid.UUID,
        status: ConversationStatus | None = None,
        subject: Subject | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Page[ConversationSummary]:
        conversations, total = await self._repository.list_conversations(
            user_id=user_id,
            status=status,
            subject=subject,
            page=page,
            page_size=page_size,
        )
        return Page[ConversationSummary](
            items=[ConversationSummary.model_validate(item) for item in conversations],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=max(1, math.ceil(total / page_size)) if total else 0,
        )
