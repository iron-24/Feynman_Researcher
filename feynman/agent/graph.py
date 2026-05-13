"""LangGraph state machine definition for the Feynman agent.

Graph topology
──────────────
Each user turn invokes the compiled graph once.  The graph routes to one
or two nodes based on state.phase and the user's message, then terminates.
State is managed externally (Gradio gr.State) so no LangGraph checkpointer
is needed; the full AgentState dict is passed in and out each turn.

Turn flow
─────────
Turn 1  │ user enters topic
        │ → research_node (fetch papers, ask assessment Qs) → END

Turn 2  │ user answers assessment
        │ → assess_node (set level) → explain_node (first concept + Q) → END

Turn 3+ │ user answers Socratic Q   → check_node → explain_node → END
        │                            (or → summarize_node → END)
        │ user asks to go deeper     → drill_node → END

Drill clarification: after drill_node the phase stays "checking", so the
next ordinary turn evaluates comprehension before resuming the concept list.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from feynman.agent.nodes import (
    assess_node,
    check_node,
    drill_node,
    explain_node,
    research_node,
    summarize_node,
)
from feynman.agent.state import AgentState

# Keywords that override normal phase routing and trigger a deep-dive
_DRILL_KEYWORDS = frozenset([
    "go deeper",
    "tell me more",
    "explain more",
    "more detail",
    "elaborate",
    "i don't understand",
    "i dont understand",
    "can you expand",
])

NodeName = Literal["research", "assess", "explain", "check", "drill", "summarize"]


def _route_entry(state: AgentState) -> NodeName:
    """Choose the first node to run for each user turn."""
    phase = state.get("phase", "start")
    user_msg = (state.get("user_message") or "").lower()

    # Drill intent overrides normal routing from any "active teaching" phase
    if phase == "checking" and any(kw in user_msg for kw in _DRILL_KEYWORDS):
        return "drill"

    mapping: dict[str, NodeName] = {
        "start":      "research",
        "assessing":  "assess",
        "checking":   "check",
        "summarizing": "summarize",
    }
    return mapping.get(phase, "research")


def _route_after_check(state: AgentState) -> NodeName:
    """After check_node, go to summarize if all concepts done, else explain."""
    if state.get("phase") == "summarizing":
        return "summarize"
    return "explain"


def build_graph() -> StateGraph:
    """Construct and compile the Feynman agent graph.

    Returns:
        A compiled LangGraph StateGraph ready for .invoke().
    """
    g = StateGraph(AgentState)

    g.add_node("research",  research_node)
    g.add_node("assess",    assess_node)
    g.add_node("explain",   explain_node)
    g.add_node("check",     check_node)
    g.add_node("drill",     drill_node)
    g.add_node("summarize", summarize_node)

    # Entry: route every turn to the correct starting node
    g.add_conditional_edges(
        START,
        _route_entry,
        {
            "research":  "research",
            "assess":    "assess",
            "explain":   "explain",
            "check":     "check",
            "drill":     "drill",
            "summarize": "summarize",
        },
    )

    # research: outputs assessment questions, then waits for user
    g.add_edge("research", END)

    # assess → explain: both run in the same turn (no user input between them)
    g.add_edge("assess", "explain")
    g.add_edge("explain", END)

    # check → explain or summarize (both run in the same turn)
    g.add_conditional_edges(
        "check",
        _route_after_check,
        {"explain": "explain", "summarize": "summarize"},
    )

    # drill: outputs deep explanation, then waits for user (phase stays "checking")
    g.add_edge("drill", END)

    g.add_edge("summarize", END)

    return g.compile()


# Module-level singleton — import this in app.py and in tests
feynman_graph = build_graph()
