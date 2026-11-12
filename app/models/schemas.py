"""Pydantic v2 request/response models for the HTTP layer."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import (
    AnswerStatus,
    ConversationMode,
    ConversationStatus,
    Difficulty,
    FallbackReason,
    MessageRole,
    NodeName,
    RoutingDecision,
    StepStatus,
    Subject,
)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
class TokenRequest(BaseModel):
    """``username`` identifies the learner; ``password`` is the shared app password."""

    username: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[\w.@+-]+$")]
    password: Annotated[str, Field(min_length=1, max_length=256)]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Seconds until the token expires")
    user_id: uuid.UUID
    username: str


# --------------------------------------------------------------------------- #
# Agent intermediate results
# --------------------------------------------------------------------------- #
class Understanding(BaseModel):
    """Output of the ``understand`` node."""

    restated_question: str
    subject: Subject
    topic: str = ""
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    knowns: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    is_answerable: bool = True


class Solution(BaseModel):
    """Output of the ``solve`` node."""

    answer: str
    steps: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class SelfCheck(BaseModel):
    """Output of the ``self_check`` node."""

    passed: bool
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    issues: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Conversations and messages
# --------------------------------------------------------------------------- #
class ConversationCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"title": "Calculus revision", "subject": "math"}}
    )

    title: Annotated[str | None, Field(default=None, max_length=200)] = None
    subject: Subject | None = None
    mode: ConversationMode = ConversationMode.TUTOR
    #: Agent mode only. Must be one of ``AGENT_MODELS``; defaults to ``AGENT_MODEL``.
    model: Annotated[str | None, Field(default=None, max_length=64)] = None


class MessageCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "A ball is dropped from 20 m. How long until it hits the ground?"
            }
        }
    )

    content: Annotated[str, Field(min_length=3, max_length=8000)]


class StepResponse(BaseModel):
    sequence: int
    node: NodeName
    status: StepStatus
    routing_decision: RoutingDecision | None = None
    latency_ms: int
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class MessageResponse(BaseModel):
    """A stored message. Assistant messages carry the agent's trace."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    role: MessageRole
    content: str
    created_at: datetime

    # Assistant-only fields.
    answer_status: AnswerStatus | None = None
    fallback_reason: FallbackReason | None = None
    confidence: float | None = None
    retry_count: int | None = None
    latency_ms: int | None = None
    error: str | None = None
    trace: list[StepResponse] = Field(default_factory=list)


class TurnResponse(BaseModel):
    """What ``POST /conversations/{id}/messages`` returns: both messages."""

    user_message: MessageResponse
    assistant_message: MessageResponse


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    subject: Subject | None = None
    status: ConversationStatus
    mode: ConversationMode = ConversationMode.TUTOR
    agent_model: str | None = None
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[MessageResponse] = Field(default_factory=list)


class Page(BaseModel, Generic[T]):
    items: list[T]
    page: int
    page_size: int
    total: int
    total_pages: int


# --------------------------------------------------------------------------- #
# Errors / health
# --------------------------------------------------------------------------- #
class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    version: str
    database: str
    llm_provider: str
    llm_model: str
    agent_backend: str | None = None
    agent_model: str | None = None
    #: "claude_login" (your subscription via the Claude CLI) or "api_key".
    agent_auth: str | None = None


# --------------------------------------------------------------------------- #
# Agent + knowledge base
# --------------------------------------------------------------------------- #
class AgentConfigResponse(BaseModel):
    """What the agent can do, for clients that render a model picker."""

    name: str
    backend: str
    auth: str
    default_model: str
    models: list[str]
    effort: str
    thinking: str
    tools: list[str]
    mcp_servers: list[str]
    max_budget_usd: float


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str | None = None
    size_bytes: int
    char_count: int
    chunk_count: int
    created_at: datetime


class DocumentSearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=2000)]
    top_k: Annotated[int | None, Field(default=None, ge=1, le=50)] = None


class DocumentSearchHit(BaseModel):
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    score: float = Field(description="Cosine similarity, higher is closer")
    content: str
