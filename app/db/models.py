"""SQLAlchemy 2.0 ORM models.

This is the structural heart of the v1 -> v2 upgrade: v1 kept the whole
conversation in ``st.session_state``, so it vanished on refresh and could not
be queried. Here users, conversations and messages are first-class rows, and
each assistant message carries the agent trace that produced it.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampType, utcnow
from app.models.enums import AnswerStatus, ConversationStatus, MessageRole


class User(Base):
    """A learner. ``username`` is the identity carried in the JWT subject."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False, default=utcnow)

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User id={self.id} username={self.username}>"


class Conversation(Base):
    """A tutoring thread owned by one user."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New conversation")
    subject: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ConversationStatus.ACTIVE, index=True
    )
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        TimestampType, nullable=False, default=utcnow, onupdate=utcnow
    )

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at, Message.sequence",
        lazy="selectin",
    )

    __table_args__ = (
        # Serves `GET /conversations` for one user, ordered newest-first.
        Index("ix_conversations_user_id_updated_at", "user_id", "updated_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Conversation id={self.id} user_id={self.user_id}>"


class Message(Base):
    """One message in a conversation.

    Assistant messages additionally store the outcome of the graph run:
    status, fallback reason, confidence, retry count, latency and the full
    node-by-node ``trace``.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # --- assistant-only run metadata -------------------------------------
    answer_status: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    fallback_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    retry_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # The node-by-node trace, stored as JSONB on PostgreSQL. Kept on the
    # message rather than in a separate table so the schema stays the three
    # entities the API exposes; see README "Design decisions".
    trace: Mapped[list[dict[str, Any]]] = mapped_column(
        Base.json_type, nullable=False, default=list
    )

    created_at: Mapped[datetime] = mapped_column(TimestampType, nullable=False, default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        UniqueConstraint("conversation_id", "sequence", name="uq_messages_conversation_sequence"),
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
    )

    @property
    def is_assistant(self) -> bool:
        return self.role == MessageRole.ASSISTANT.value

    @property
    def is_verified(self) -> bool:
        return self.answer_status == AnswerStatus.VERIFIED.value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Message id={self.id} role={self.role} seq={self.sequence}>"
