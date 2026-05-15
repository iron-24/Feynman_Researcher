"""Citation grounding checker for Feynman agent explanations.

For every substantive sentence in an explanation, searches the vector store
for a supporting chunk. Reports a grounding rate (% of sentences backed by
a paper excerpt) and flags unsupported sentences as potential hallucinations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import structlog

from feynman.retrieval.vector_store import VectorStore

log = structlog.get_logger()

GROUNDING_THRESHOLD = 0.40  # cosine similarity — tune up to reduce false negatives
MIN_WORDS = 8               # skip very short sentences ("Yes.", "OK.", etc.)


@dataclass
class CitationCheckResult:
    """Result of a single citation grounding check."""

    total_sentences: int
    grounded: int
    ungrounded: List[str] = field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        return self.grounded / self.total_sentences if self.total_sentences else 1.0

    @property
    def hallucination_rate(self) -> float:
        return 1.0 - self.grounding_rate

    def summary(self) -> str:
        return (
            f"Grounding: {self.grounding_rate:.1%}  "
            f"({self.grounded}/{self.total_sentences} sentences supported)\n"
            f"Hallucination rate: {self.hallucination_rate:.1%}"
        )


def check_citations(explanation: str, store: VectorStore) -> CitationCheckResult:
    """Check how many sentences in an explanation are grounded in the vector store.

    Each sentence is embedded and searched against the FAISS index. A sentence
    is considered grounded if its nearest neighbour has cosine similarity ≥
    GROUNDING_THRESHOLD (meaning a paper excerpt exists that says something
    semantically similar).

    Args:
        explanation: Text output from explain_node or re_explain_node.
        store: Populated VectorStore for the current session.

    Returns:
        CitationCheckResult with per-sentence breakdown.
    """
    sentences = _split_sentences(explanation)
    grounded = 0
    ungrounded: List[str] = []

    for sentence in sentences:
        results = store.search(sentence, k=1)
        if results and results[0].score >= GROUNDING_THRESHOLD:
            grounded += 1
        else:
            ungrounded.append(sentence)

    result = CitationCheckResult(
        total_sentences=len(sentences),
        grounded=grounded,
        ungrounded=ungrounded,
    )
    log.info(
        "citation_check_done",
        total=result.total_sentences,
        grounded=grounded,
        hallucination_rate=f"{result.hallucination_rate:.1%}",
    )
    return result


def _split_sentences(text: str) -> List[str]:
    """Strip markdown formatting and return substantive sentences."""
    clean = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # bold / italic
    clean = re.sub(r"`[^`]+`", "", clean)                    # inline code
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)  # links

    raw = re.split(r"(?<=[.!?])\s+", clean.strip())
    return [s.strip() for s in raw if len(s.split()) >= MIN_WORDS]
