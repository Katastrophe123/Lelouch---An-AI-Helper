"""Assembly of the tutoring state graph.

    understand ──not answerable / error──────────────────────┐
        │                                                    │
        └─ok─> solve ──> self_check                          │
                 ▲            │                              │
                 └─retry ≤N───┤                              │
                              ├─passed──> explain ──> END    │
                              └─failed──> fallback <─────────┘──> END

Routers are pure functions of ``state["last_decision"]``, which each node sets
alongside the step record it persists - so what the graph did and why is always
recoverable from the stored message.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import Emitter, NodeConfig, TutorNodes
from app.agents.state import TutorState
from app.config import Settings
from app.llm.base import LLMClient
from app.llm.retry import SleepFn
from app.models.enums import NodeName, RoutingDecision

FALLBACK_DECISIONS: frozenset[str] = frozenset(
    {
        RoutingDecision.TO_FALLBACK_CHECK_FAILED.value,
        RoutingDecision.TO_FALLBACK_NODE_ERROR.value,
        RoutingDecision.TO_FALLBACK_STEP_LIMIT.value,
        RoutingDecision.TO_FALLBACK_NOT_ANSWERABLE.value,
    }
)


def route_after_understand(state: TutorState) -> str:
    """``understand`` -> ``solve`` unless the question is unusable."""
    if state.get("last_decision") in FALLBACK_DECISIONS:
        return NodeName.FALLBACK.value
    return NodeName.SOLVE.value


def route_after_solve(state: TutorState) -> str:
    """``solve`` -> ``self_check`` unless solving failed outright."""
    if state.get("last_decision") in FALLBACK_DECISIONS:
        return NodeName.FALLBACK.value
    return NodeName.SELF_CHECK.value


def route_after_self_check(state: TutorState) -> str:
    """Three-way branch: explain, retry the solve, or fall back."""
    decision = state.get("last_decision")
    if decision == RoutingDecision.TO_EXPLAIN.value:
        return NodeName.EXPLAIN.value
    if decision == RoutingDecision.RETRY_SOLVE.value:
        return NodeName.SOLVE.value
    return NodeName.FALLBACK.value


def route_after_explain(state: TutorState) -> str:
    """``explain`` normally ends the turn, but can still fall back on error."""
    if state.get("last_decision") in FALLBACK_DECISIONS:
        return NodeName.FALLBACK.value
    return END


def build_graph(nodes: TutorNodes) -> Any:
    """Wire ``nodes`` into a compiled LangGraph application."""
    builder: StateGraph = StateGraph(TutorState)

    builder.add_node(NodeName.UNDERSTAND.value, nodes.understand)
    builder.add_node(NodeName.SOLVE.value, nodes.solve)
    builder.add_node(NodeName.SELF_CHECK.value, nodes.self_check)
    builder.add_node(NodeName.EXPLAIN.value, nodes.explain)
    builder.add_node(NodeName.FALLBACK.value, nodes.fallback)

    builder.add_edge(START, NodeName.UNDERSTAND.value)

    builder.add_conditional_edges(
        NodeName.UNDERSTAND.value,
        route_after_understand,
        {
            NodeName.SOLVE.value: NodeName.SOLVE.value,
            NodeName.FALLBACK.value: NodeName.FALLBACK.value,
        },
    )
    builder.add_conditional_edges(
        NodeName.SOLVE.value,
        route_after_solve,
        {
            NodeName.SELF_CHECK.value: NodeName.SELF_CHECK.value,
            NodeName.FALLBACK.value: NodeName.FALLBACK.value,
        },
    )
    builder.add_conditional_edges(
        NodeName.SELF_CHECK.value,
        route_after_self_check,
        {
            NodeName.EXPLAIN.value: NodeName.EXPLAIN.value,
            NodeName.SOLVE.value: NodeName.SOLVE.value,
            NodeName.FALLBACK.value: NodeName.FALLBACK.value,
        },
    )
    builder.add_conditional_edges(
        NodeName.EXPLAIN.value,
        route_after_explain,
        {NodeName.FALLBACK.value: NodeName.FALLBACK.value, END: END},
    )

    builder.add_edge(NodeName.FALLBACK.value, END)

    return builder.compile()


def build_tutor_graph(
    *,
    settings: Settings,
    llm: LLMClient,
    sleep: SleepFn | None = None,
    emitter: Emitter | None = None,
) -> Any:
    """Convenience factory.

    The API builds one graph per process for the non-streaming endpoint, and a
    fresh graph per request for the streaming one (so the emitter can be bound
    to that request's event queue).
    """
    nodes = TutorNodes(
        llm=llm,
        config=NodeConfig.from_settings(settings),
        sleep=sleep,
        emitter=emitter,
    )
    return build_graph(nodes)
