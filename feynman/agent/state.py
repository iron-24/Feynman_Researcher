"""AgentState — the single source of truth that flows through the LangGraph nodes."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    """Full runtime state of the Feynman agent.

    total=False makes every field optional so nodes can return partial updates
    via dict and LangGraph will merge them into the running state.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    topic: str          # the subject the user wants to learn
    thread_id: str      # ties this session to a Gradio state object

    # ── Retrieved knowledge ───────────────────────────────────────────────
    papers: List[Any]                       # List[Paper] — kept as Any to avoid circular import
    vector_store: Any                       # VectorStore instance (in-memory FAISS)
    knowledge_graph: Dict[str, List[str]]   # concept → list of prerequisite concepts
    concepts_ordered: List[str]             # teaching sequence, foundational → advanced
    concepts_covered: List[str]             # concepts the user has demonstrated understanding of

    # ── Teaching state ────────────────────────────────────────────────────
    current_concept: str    # concept being explained right now
    user_level: str         # "beginner" | "intermediate" | "advanced"
    reexplain: bool         # True → explain_node uses a different framing for same concept
    pending_questions: List[str]  # Socratic questions asked but not yet answered

    # ── Conversation ──────────────────────────────────────────────────────
    conversation_history: List[Dict[str, str]]  # [{"role": "user"|"assistant", "content": str}]
    user_message: str    # latest message received from the user
    agent_response: str  # latest message produced by the agent

    # ── Flow control ──────────────────────────────────────────────────────
    # Lifecycle:  start → assessing → checking (loops) → summarizing → done
    phase: str
