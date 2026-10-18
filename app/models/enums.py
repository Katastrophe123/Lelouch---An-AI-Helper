"""Domain enumerations shared by the graph, the database and the API."""

from __future__ import annotations

from enum import StrEnum


class Subject(StrEnum):
    """Academic subject inferred by the ``understand`` node."""

    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    COMPUTER_SCIENCE = "computer_science"
    ECONOMICS = "economics"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    ENGLISH = "english"
    OTHER = "other"


class Difficulty(StrEnum):
    INTRODUCTORY = "introductory"
    INTERMEDIATE = "advanced_school"
    UNIVERSITY = "university"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AnswerStatus(StrEnum):
    """Terminal state of one tutoring turn."""

    PENDING = "pending"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class StepStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class NodeName(StrEnum):
    """Graph node identifiers, also used as step labels in the trace."""

    UNDERSTAND = "understand"
    SOLVE = "solve"
    SELF_CHECK = "self_check"
    EXPLAIN = "explain"
    FALLBACK = "fallback"


class RoutingDecision(StrEnum):
    """Outcome of a conditional edge, persisted for auditability."""

    TO_SOLVE = "to_solve"
    TO_SELF_CHECK = "to_self_check"
    TO_EXPLAIN = "to_explain"
    RETRY_SOLVE = "retry_solve"
    TO_FALLBACK_CHECK_FAILED = "to_fallback_check_failed"
    TO_FALLBACK_NODE_ERROR = "to_fallback_node_error"
    TO_FALLBACK_STEP_LIMIT = "to_fallback_step_limit"
    TO_FALLBACK_NOT_ANSWERABLE = "to_fallback_not_answerable"
    END = "end"


class FallbackReason(StrEnum):
    """Why a turn returned an unverified answer instead of a checked one."""

    SELF_CHECK_FAILED = "self_check_failed"
    NODE_ERROR = "node_error"
    STEP_LIMIT_REACHED = "step_limit_reached"
    NOT_ANSWERABLE = "not_answerable"


class StreamEventType(StrEnum):
    """Server-Sent Event names emitted by the streaming endpoint."""

    STEP = "step"
    TOKEN = "token"
    DONE = "done"
    ERROR = "error"
