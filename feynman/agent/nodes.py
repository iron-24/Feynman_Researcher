"""LangGraph node functions for the Feynman agent.

Each node is a pure function: (AgentState) -> dict (partial state update).
Nodes never mutate state in-place — they return only the keys they change.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import structlog
from dotenv import load_dotenv
from google import genai
from google.genai import types

from feynman.agent.prompts import (
    ASSESS_KNOWLEDGE,
    CHECK_UNDERSTANDING,
    CONCEPT_EXTRACTION,
    DRILL_DEEP,
    EXPLAIN_CONCEPT,
    INFER_LEVEL,
    RE_EXPLAIN_CONCEPT,
    SUMMARIZE_LEARNING,
)
from feynman.agent.state import AgentState
from feynman.retrieval.arxiv_client import ArXivClient
from feynman.retrieval.pdf_parser import PDFParser
from feynman.retrieval.semantic_scholar import SemanticScholarClient
from feynman.retrieval.vector_store import VectorStore, SearchResult

load_dotenv()
log = structlog.get_logger()

MODEL = "models/gemini-3.1-flash-lite"
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


# ══════════════════════════════════════════════════════════════════════════════
# research_node
# ══════════════════════════════════════════════════════════════════════════════

def research_node(state: AgentState) -> dict:
    """Fetch papers, build the vector store, extract ordered concepts, ask assessment Qs.

    This is the only node that hits the network heavily. It:
    1. Searches ArXiv + enriches with Semantic Scholar citation counts
    2. Parses and indexes the top-10 PDFs
    3. Asks Claude to order concepts from foundational to advanced
    4. Returns assessment questions so the next turn can calibrate to the user's level
    """
    topic = state.get("topic") or state.get("user_message", "")
    log.info("research_start", topic=topic)

    # 1. Fetch and rank papers
    papers = ArXivClient(years_back=5).search(topic, max_results=15)
    papers = SemanticScholarClient().enrich_papers(papers)
    papers.sort(key=lambda p: p.citation_count, reverse=True)

    # 2. Parse PDFs and build vector store
    parser = PDFParser()
    store = VectorStore()
    indexed = 0
    for paper in papers[:10]:
        if not paper.pdf_url:
            continue
        chunks = parser.parse(paper.id, paper.title, paper.pdf_url)
        if chunks:
            store.add_paper(chunks)
            indexed += 1
            log.info("indexed", title=paper.title[:60], chunks=len(chunks))

    # 3. Extract ordered concept list from abstracts
    abstracts = "\n\n".join(
        f"[{p.title}]\n{p.abstract}" for p in papers[:8]
    )
    raw = _call_claude(
        system="You are a curriculum designer for a research-paper-based learning system.",
        user=CONCEPT_EXTRACTION.format(topic=topic, abstracts=abstracts),
        temperature=0.3,
    )
    concept_data = _parse_json(raw)
    concepts_ordered: List[str] = concept_data.get("concepts", [topic])
    knowledge_graph: Dict[str, List[str]] = concept_data.get("knowledge_graph", {})

    # 4. Ask assessment questions
    assessment = _call_claude(
        system="You are a helpful and encouraging research-paper-based tutor.",
        user=ASSESS_KNOWLEDGE.format(topic=topic),
    )

    log.info("research_done", papers=len(papers), indexed=indexed, concepts=len(concepts_ordered))
    return {
        "topic": topic,
        "papers": papers,
        "vector_store": store,
        "knowledge_graph": knowledge_graph,
        "concepts_ordered": concepts_ordered,
        "concepts_covered": [],
        "pending_questions": [],
        "conversation_history": [{"role": "assistant", "content": assessment}],
        "agent_response": assessment,
        "phase": "assessing",
    }


# ══════════════════════════════════════════════════════════════════════════════
# assess_node
# ══════════════════════════════════════════════════════════════════════════════

def assess_node(state: AgentState) -> dict:
    """Process the user's assessment answers and set their level.

    This node runs in the same graph turn as explain_node — it only updates
    state so explain_node has the right user_level and current_concept.
    No agent_response is set here; explain_node provides the output.
    """
    topic = state["topic"]
    user_message = state.get("user_message", "")
    concepts_ordered: List[str] = state.get("concepts_ordered", [topic])

    raw = _call_claude(
        system="You are evaluating a student's background knowledge concisely.",
        user=INFER_LEVEL.format(topic=topic, user_response=user_message),
        temperature=0.2,
    )
    level_data = _parse_json(raw)
    user_level: str = level_data.get("user_level", "beginner")

    # Skip ahead in the concept list for more experienced learners
    start_idx = {"beginner": 0, "intermediate": 1, "advanced": 2}.get(user_level, 0)
    start_idx = min(start_idx, len(concepts_ordered) - 1)
    current_concept = concepts_ordered[start_idx]

    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": user_message})

    log.info("assessed", level=user_level, starting=current_concept)
    return {
        "user_level": user_level,
        "current_concept": current_concept,
        "reexplain": False,
        "conversation_history": history,
    }


# ══════════════════════════════════════════════════════════════════════════════
# explain_node
# ══════════════════════════════════════════════════════════════════════════════

def explain_node(state: AgentState) -> dict:
    """Explain the current concept using paper-grounded RAG and ask a Socratic question.

    If state.reexplain is True, uses RE_EXPLAIN_CONCEPT with a different framing.
    If all concepts are already covered, short-circuits to trigger summarization.
    """
    concepts_covered: List[str] = state.get("concepts_covered", [])
    concepts_ordered: List[str] = state.get("concepts_ordered", [])
    user_level: str = state.get("user_level", "beginner")
    reexplain: bool = state.get("reexplain", False)
    store: Optional[VectorStore] = state.get("vector_store")

    # Determine which concept to explain
    if reexplain:
        concept = state.get("current_concept", "")
    else:
        remaining = [c for c in concepts_ordered if c not in concepts_covered]
        if not remaining:
            # Nothing left to explain — hand off to summarize_node
            return {"phase": "summarizing"}
        concept = remaining[0]

    context = _get_context(store, concept) if store else "No paper context available."
    gaps = state.get("_gaps", [])

    if reexplain:
        text = _call_claude(
            system="You are a patient Feynman teacher. Use a completely different analogy.",
            user=RE_EXPLAIN_CONCEPT.format(
                concept=concept,
                gaps=", ".join(gaps) if gaps else "general confusion",
                user_level=user_level,
                context=context,
            ),
        )
    else:
        covered_str = ", ".join(concepts_covered) if concepts_covered else "none yet"
        text = _call_claude(
            system="You are a Feynman-style teacher. Every claim must cite a paper by title.",
            user=EXPLAIN_CONCEPT.format(
                concept=concept,
                topic=state.get("topic", ""),
                user_level=user_level,
                concepts_covered=covered_str,
                context=context,
            ),
        )

    question = _extract_last_question(text)

    # Prepend level badge only on the very first explanation (after assessment)
    history = list(state.get("conversation_history", []))
    if not concepts_covered and not reexplain:
        badge = f"**Level set: {user_level}**\n\n"
        response = badge + text
    else:
        response = text

    history.append({"role": "assistant", "content": response})

    return {
        "current_concept": concept,
        "reexplain": False,
        "conversation_history": history,
        "pending_questions": [question] if question else ["What's your main takeaway?"],
        "agent_response": response,
        "phase": "checking",
    }


# ══════════════════════════════════════════════════════════════════════════════
# check_node
# ══════════════════════════════════════════════════════════════════════════════

def check_node(state: AgentState) -> dict:
    """Evaluate the user's answer and update teaching state.

    After evaluation, this node sets up state for explain_node (which runs
    next in the same graph turn):
    - understood=True  → mark concept covered, current_concept = next, reexplain=False
    - understood=False → keep current_concept, reexplain=True, store gaps
    - all concepts done → phase="summarizing" (routes to summarize_node instead)
    """
    topic = state["topic"]
    user_message = state.get("user_message", "")
    current_concept = state.get("current_concept", "")
    user_level = state.get("user_level", "beginner")
    concepts_covered = list(state.get("concepts_covered", []))
    concepts_ordered = state.get("concepts_ordered", [])
    pending_questions = state.get("pending_questions", [])

    question = pending_questions[-1] if pending_questions else "What is your takeaway?"

    raw = _call_claude(
        system="You are rigorously but fairly evaluating a student's conceptual understanding.",
        user=CHECK_UNDERSTANDING.format(
            concept=current_concept,
            user_level=user_level,
            question=question,
            user_answer=user_message,
        ),
        temperature=0.2,
    )
    result = _parse_json(raw)
    understood: bool = result.get("understood", False)
    gaps: List[str] = result.get("gaps", [])
    praise: str = result.get("praise", "Good effort!")

    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": user_message})

    if understood:
        if current_concept not in concepts_covered:
            concepts_covered.append(current_concept)
        remaining = [c for c in concepts_ordered if c not in concepts_covered]

        if not remaining:
            # All done — route to summarize_node
            transition = f"{praise}\n\nYou've covered all the core concepts! Let me pull it all together..."
            history.append({"role": "assistant", "content": transition})
            return {
                "concepts_covered": concepts_covered,
                "conversation_history": history,
                "agent_response": transition,
                "phase": "summarizing",
            }

        # Advance to next concept — explain_node will handle the explanation
        next_concept = remaining[0]
        log.info("concept_understood", concept=current_concept, next=next_concept)
        return {
            "concepts_covered": concepts_covered,
            "current_concept": next_concept,
            "reexplain": False,
            "_gaps": [],
            "conversation_history": history,
            "_check_praise": praise,
            "phase": "checking",  # stays checking; explain_node will set it back
        }

    else:
        # Not yet understood — explain_node will re-explain
        log.info("concept_not_understood", concept=current_concept, gaps=gaps)
        return {
            "reexplain": True,
            "_gaps": gaps,
            "conversation_history": history,
            "_check_praise": praise,
            "phase": "checking",
        }


# ══════════════════════════════════════════════════════════════════════════════
# drill_node
# ══════════════════════════════════════════════════════════════════════════════

def drill_node(state: AgentState) -> dict:
    """Go deeper on a subtopic the user asked about.

    Fetches additional targeted papers, expands the vector store, then
    produces a paper-grounded deep dive. After drilling, phase stays
    "checking" so the next turn evaluates comprehension before resuming
    the main concept sequence.
    """
    topic = state["topic"]
    user_message = state.get("user_message", "")
    user_level = state.get("user_level", "beginner")
    concepts_covered = state.get("concepts_covered", [])
    store: Optional[VectorStore] = state.get("vector_store")
    history = list(state.get("conversation_history", []))

    # Fetch targeted papers for the subtopic
    extra = ArXivClient(years_back=5).search(f"{topic} {user_message}", max_results=5)
    if extra and store:
        parser = PDFParser()
        for paper in extra[:3]:
            if paper.pdf_url:
                chunks = parser.parse(paper.id, paper.title, paper.pdf_url)
                if chunks:
                    store.add_paper(chunks)
                    log.info("drill_paper_indexed", title=paper.title[:60])

    context = _get_context(store, f"{topic} {user_message}", k=7) if store else ""
    covered_str = ", ".join(concepts_covered) if concepts_covered else "none yet"

    response = _call_claude(
        system="You are a research expert going deep on a subtopic. Cite papers directly.",
        user=DRILL_DEEP.format(
            subtopic=user_message,
            topic=topic,
            user_level=user_level,
            concepts_covered=covered_str,
            context=context,
        ),
    )

    question = _extract_last_question(response)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})

    return {
        "vector_store": store,
        "conversation_history": history,
        "pending_questions": [question] if question else state.get("pending_questions", []),
        "agent_response": response,
        "phase": "checking",
    }


# ══════════════════════════════════════════════════════════════════════════════
# summarize_node
# ══════════════════════════════════════════════════════════════════════════════

def summarize_node(state: AgentState) -> dict:
    """Generate the Feynman summary — the proof that learning happened.

    Uses the last 10 conversation turns as context to ground the summary
    in the analogies and framings that actually worked for this learner.
    """
    topic = state["topic"]
    concepts_covered = state.get("concepts_covered", [])
    user_level = state.get("user_level", "beginner")
    history = state.get("conversation_history", [])

    conv_summary = "\n".join(
        f"{m['role'].upper()}: {m['content'][:400]}"
        for m in history[-10:]
    )

    summary = _call_claude(
        system="You are writing the definitive Feynman summary of a learning session. Be clear, elegant, complete.",
        user=SUMMARIZE_LEARNING.format(
            topic=topic,
            concepts_covered=", ".join(concepts_covered) if concepts_covered else topic,
            user_level=user_level,
            conversation_summary=conv_summary,
        ),
        temperature=0.5,
    )

    history_updated = list(history)
    history_updated.append({"role": "assistant", "content": summary})

    return {
        "conversation_history": history_updated,
        "agent_response": summary,
        "phase": "done",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers (module-private)
# ══════════════════════════════════════════════════════════════════════════════

def _call_claude(system: str, user: str, temperature: float = 0.7) -> str:
    """Single-turn Gemini call. Returns the model's text response."""
    try:
        resp = _get_client().models.generate_content(
            model=MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        return resp.text
    except Exception as exc:
        msg = str(exc)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise RuntimeError(
                "You have exceeded your Gemini API quota. Check your usage at https://ai.dev/rate-limit."
            ) from None
        if "503" in msg or "UNAVAILABLE" in msg:
            raise RuntimeError(
                "Gemini is temporarily overloaded — please wait a few seconds and try again."
            ) from None
        raise RuntimeError(f"Gemini API error: {exc}") from None


def _parse_json(text: str) -> Dict:
    """Extract and parse JSON from a Claude response, stripping markdown fences."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        log.warning("json_parse_failed", raw=text[:200])
        return {}


def _get_context(store: VectorStore, query: str, k: int = 5) -> str:
    """Retrieve top-k chunks and format them as a prompt-ready string."""
    results: List[SearchResult] = store.search(query, k=k)
    if not results:
        return "No relevant paper excerpts found."
    return "\n\n---\n\n".join(
        f"[{r.chunk.title} — {r.chunk.section}]\n{r.chunk.text}"
        for r in results
    )


def _extract_last_question(text: str) -> str:
    """Return the last sentence ending in '?' from a block of text."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in reversed(sentences):
        if sentence.strip().endswith("?"):
            return sentence.strip()
    return ""
