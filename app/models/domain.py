"""Internal domain objects shared between the agent graph and persistence.

Separate from ``app.models.schemas`` (the HTTP contract) and from
``app.db.models`` (SQLAlchemy), so the graph depends on neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.enums import MessageRole, NodeName, RoutingDecision, StepStatus


@dataclass(slots=True)
class StepRecord:
    """Audit record for one graph node execution."""

    sequence: int
    node: NodeName
    status: StepStatus
    latency_ms: int
    output: dict[str, Any] = field(default_factory=dict)
    routing_decision: RoutingDecision | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "node": self.node.value,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "output": self.output,
            "routing_decision": (self.routing_decision.value if self.routing_decision else None),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepRecord:
        routing = data.get("routing_decision")
        return cls(
            sequence=int(data["sequence"]),
            node=NodeName(data["node"]),
            status=StepStatus(data["status"]),
            latency_ms=int(data.get("latency_ms", 0)),
            output=dict(data.get("output") or {}),
            routing_decision=RoutingDecision(routing) if routing else None,
            error=data.get("error"),
        )


@dataclass(frozen=True, slots=True)
class HistoryTurn:
    """One prior message, replayed to the model as conversation context."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryTurn:
        return cls(role=MessageRole(data["role"]), content=str(data["content"]))
