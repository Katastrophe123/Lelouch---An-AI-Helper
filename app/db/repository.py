"""Data-access layer for users, conversations and messages.

All SQL lives here so services stay free of query construction.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utcnow
from app.db.models import Conversation, Document, DocumentChunk, Message, User
from app.models.domain import HistoryTurn, StepRecord
from app.models.enums import (
    AnswerStatus,
    ConversationMode,
    ConversationStatus,
    FallbackReason,
    MessageRole,
    Subject,
)


class UserRepository:
    """Lookup and get-or-create for learners."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_username(self, username: str) -> User | None:
        stmt = select(User).where(User.username == username)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        stmt = select(User).where(User.id == user_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_or_create(self, username: str, *, display_name: str | None = None) -> User:
        """Return the user for ``username``, creating the row on first sign-in."""
        existing = await self.get_by_username(username)
        if existing is not None:
            return existing
        user = User(username=username, display_name=display_name or username)
        self._session.add(user)
        await self._session.flush()
        return user


class ConversationRepository:
    """CRUD and queries for conversations and their messages."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # --- conversations ----------------------------------------------------
    async def create_conversation(
        self,
        *,
        user_id: uuid.UUID,
        title: str | None = None,
        subject: Subject | None = None,
        mode: ConversationMode = ConversationMode.TUTOR,
        agent_model: str | None = None,
    ) -> Conversation:
        conversation = Conversation(
            user_id=user_id,
            title=(title or "New conversation")[:200],
            subject=subject.value if subject else None,
            status=ConversationStatus.ACTIVE.value,
            mode=mode.value,
            agent_model=agent_model if mode is ConversationMode.AGENT else None,
            message_count=0,
        )
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def get_conversation(
        self, conversation_id: uuid.UUID, *, user_id: uuid.UUID | None = None
    ) -> Conversation | None:
        """Fetch a conversation with its messages.

        Passing ``user_id`` scopes the lookup to that owner, so a caller can
        never read another learner's thread by guessing an id.
        """
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        if user_id is not None:
            stmt = stmt.where(Conversation.user_id == user_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    @staticmethod
    def _apply_conversation_filters(
        stmt: Select[Any],
        *,
        user_id: uuid.UUID,
        status: ConversationStatus | None,
        subject: Subject | None,
    ) -> Select[Any]:
        stmt = stmt.where(Conversation.user_id == user_id)
        if status is not None:
            stmt = stmt.where(Conversation.status == status.value)
        if subject is not None:
            stmt = stmt.where(Conversation.subject == subject.value)
        return stmt

    async def list_conversations(
        self,
        *,
        user_id: uuid.UUID,
        status: ConversationStatus | None = None,
        subject: Subject | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[Sequence[Conversation], int]:
        """Return ``(page_of_conversations, total)`` most-recently-updated first."""
        count_stmt = self._apply_conversation_filters(
            select(Conversation).with_only_columns(func.count(Conversation.id)).order_by(None),
            user_id=user_id,
            status=status,
            subject=subject,
        )
        total = (await self._session.execute(count_stmt)).scalar_one()

        page_stmt = self._apply_conversation_filters(
            select(Conversation), user_id=user_id, status=status, subject=subject
        )
        page_stmt = (
            page_stmt.order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        rows = (await self._session.execute(page_stmt)).scalars().all()
        return rows, int(total)

    async def touch_conversation(
        self, conversation: Conversation, *, subject: Subject | None = None
    ) -> Conversation:
        """Bump ``updated_at`` and backfill the subject from the first answer."""
        conversation.updated_at = utcnow()
        if subject is not None and conversation.subject is None:
            conversation.subject = subject.value
        await self._session.flush()
        return conversation

    # --- messages ---------------------------------------------------------
    async def next_sequence(self, conversation_id: uuid.UUID) -> int:
        stmt = select(func.coalesce(func.max(Message.sequence), 0)).where(
            Message.conversation_id == conversation_id
        )
        current = (await self._session.execute(stmt)).scalar_one()
        return int(current) + 1

    async def add_user_message(self, *, conversation: Conversation, content: str) -> Message:
        message = Message(
            conversation_id=conversation.id,
            sequence=await self.next_sequence(conversation.id),
            role=MessageRole.USER.value,
            content=content,
            trace=[],
        )
        self._session.add(message)
        conversation.message_count += 1
        await self._session.flush()
        return message

    async def add_assistant_message(
        self,
        *,
        conversation: Conversation,
        content: str,
        answer_status: AnswerStatus,
        fallback_reason: FallbackReason | None = None,
        confidence: float | None = None,
        retry_count: int = 0,
        latency_ms: int | None = None,
        error: str | None = None,
        trace: Sequence[StepRecord] = (),
    ) -> Message:
        message = Message(
            conversation_id=conversation.id,
            sequence=await self.next_sequence(conversation.id),
            role=MessageRole.ASSISTANT.value,
            content=content,
            answer_status=answer_status.value,
            fallback_reason=fallback_reason.value if fallback_reason else None,
            confidence=confidence,
            retry_count=retry_count,
            latency_ms=latency_ms,
            error=error,
            verified=answer_status is AnswerStatus.VERIFIED,
            trace=[record.to_dict() for record in trace],
        )
        self._session.add(message)
        conversation.message_count += 1
        await self._session.flush()
        return message

    async def recent_history(self, conversation_id: uuid.UUID, *, limit: int) -> list[HistoryTurn]:
        """Return the last ``limit`` messages, oldest-first, for model context."""
        if limit <= 0:
            return []
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence.desc())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [
            HistoryTurn(role=MessageRole(row.role), content=row.content) for row in reversed(rows)
        ]

    async def refresh_messages(self, conversation: Conversation) -> Conversation:
        """Load ``conversation.messages`` explicitly.

        Messages are inserted by ``conversation_id`` rather than through the
        relationship, so the collection is still unloaded; touching it lazily
        under asyncio would raise ``MissingGreenlet``.
        """
        await self._session.refresh(conversation, attribute_names=["messages"])
        return conversation

    async def count_messages_by_status(self, user_id: uuid.UUID) -> dict[str, int]:
        """Aggregate of assistant answer statuses, used by the Streamlit sidebar."""
        stmt = (
            select(Message.answer_status, func.count(Message.id))
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(Conversation.user_id == user_id, Message.answer_status.is_not(None))
            .group_by(Message.answer_status)
        )
        rows = await self._session.execute(stmt)
        return {status: int(count) for status, count in rows.all() if status}


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ChunkHit:
    """One retrieved passage with its cosine similarity to the query."""

    document_id: uuid.UUID
    filename: str
    chunk_index: int
    content: str
    score: float


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class DocumentRepository:
    """Documents, their embedded chunks, and user-scoped similarity search."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_document(
        self,
        *,
        user_id: uuid.UUID,
        filename: str,
        content_type: str | None,
        size_bytes: int,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> Document:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        document = Document(
            user_id=user_id,
            filename=filename[:255],
            content_type=content_type,
            size_bytes=size_bytes,
            char_count=sum(len(chunk) for chunk in chunks),
            chunk_count=len(chunks),
        )
        self._session.add(document)
        await self._session.flush()
        self._session.add_all(
            DocumentChunk(
                document_id=document.id,
                user_id=user_id,
                chunk_index=index,
                content=chunk,
                embedding=[float(value) for value in vector],
            )
            for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True))
        )
        await self._session.flush()
        return document

    async def list_documents(self, *, user_id: uuid.UUID) -> Sequence[Document]:
        stmt = (
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc(), Document.id.desc())
        )
        return (await self._session.execute(stmt)).scalars().all()

    async def get_document(self, document_id: uuid.UUID, *, user_id: uuid.UUID) -> Document | None:
        stmt = select(Document).where(Document.id == document_id, Document.user_id == user_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def delete_document(self, document_id: uuid.UUID, *, user_id: uuid.UUID) -> bool:
        document = await self.get_document(document_id, user_id=user_id)
        if document is None:
            return False
        # Delete chunks explicitly: SQLite does not enforce ON DELETE CASCADE
        # unless foreign keys are switched on, and the tests run on SQLite.
        chunks = await self._session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        for chunk in chunks.scalars():
            await self._session.delete(chunk)
        await self._session.delete(document)
        await self._session.flush()
        return True

    async def search(
        self, *, user_id: uuid.UUID, embedding: Sequence[float], top_k: int
    ) -> list[ChunkHit]:
        """Nearest chunks to ``embedding`` among this user's documents only."""
        dialect = self._session.get_bind().dialect.name
        if dialect == "postgresql":
            distance = DocumentChunk.embedding.cosine_distance(list(embedding)).label("distance")
            stmt = (
                select(DocumentChunk, Document.filename, distance)
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(DocumentChunk.user_id == user_id)
                .order_by(distance)
                .limit(top_k)
            )
            rows = (await self._session.execute(stmt)).all()
            return [
                ChunkHit(
                    document_id=chunk.document_id,
                    filename=filename,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=1.0 - float(dist),
                )
                for chunk, filename, dist in rows
            ]

        # Portable fallback (SQLite in tests): score in Python.
        stmt = (
            select(DocumentChunk, Document.filename)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.user_id == user_id)
        )
        scored = [
            ChunkHit(
                document_id=chunk.document_id,
                filename=filename,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=cosine_similarity(embedding, chunk.embedding),
            )
            for chunk, filename in (await self._session.execute(stmt)).all()
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]
