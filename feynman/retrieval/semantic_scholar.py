"""Semantic Scholar API wrapper — enriches papers with citation data."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
import structlog

from feynman.retrieval.arxiv_client import Paper

log = structlog.get_logger()

_BASE = "https://api.semanticscholar.org/graph/v1"
_PAPER_FIELDS = "paperId,title,authors,year,citationCount,externalIds"
_RELATED_FIELDS = "paperId,title,authors,year,citationCount,externalIds"
_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0


class SemanticScholarClient:
    """Enriches ArXiv papers with citation counts and surface related work.

    Free tier rate limit: 100 requests / 5 minutes.
    Automatic exponential backoff on HTTP 429.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich_papers(self, papers: List[Paper]) -> List[Paper]:
        """Add citation_count and semantic_scholar_id to ArXiv papers in-place.

        Matches via the ArXiv external ID (`arXiv:{id}`) so no extra search
        is needed. Papers that Semantic Scholar doesn't know about keep their
        default citation_count=0.

        Args:
            papers: Papers returned by ArXivClient.search().

        Returns:
            The same list, mutated with enriched citation data.
        """
        for paper in papers:
            arxiv_id = paper.id.split("v")[0]
            data = self._get_paper(f"arXiv:{arxiv_id}")
            if data:
                paper.citation_count = data.get("citationCount") or 0
                paper.semantic_scholar_id = data.get("paperId")
            log.debug("enriched", paper_id=paper.id, citations=paper.citation_count)
        return papers

    def get_related(self, paper: Paper, limit: int = 25) -> List[Paper]:
        """Return papers that cite or are referenced by *paper*.

        Combines results from both the /citations and /references endpoints,
        deduplicates by ArXiv ID, and returns only papers that have an ArXiv
        ID (so they can later be fetched and parsed).

        Args:
            paper: A Paper with a populated semantic_scholar_id.
            limit: Per-endpoint result cap.

        Returns:
            List of related Paper objects with citation counts.
        """
        ss_id = paper.semantic_scholar_id
        if not ss_id:
            log.warning("no_ss_id_for_related", title=paper.title)
            return []

        seen: Dict[str, Paper] = {}
        for endpoint in (
            f"/paper/{ss_id}/citations",
            f"/paper/{ss_id}/references",
        ):
            data = self._get(
                endpoint,
                params={"fields": _RELATED_FIELDS, "limit": limit},
            )
            if not data:
                continue
            for item in data.get("data", []):
                raw = item.get("citedPaper") or item.get("citingPaper") or {}
                converted = self._to_paper(raw)
                if converted and converted.id not in seen:
                    seen[converted.id] = converted

        return list(seen.values())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_paper(self, paper_id: str) -> Optional[Dict]:
        return self._get(f"/paper/{paper_id}", params={"fields": _PAPER_FIELDS})

    def _get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        url = _BASE + path
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    wait = _BACKOFF_BASE**attempt
                    log.warning("rate_limited", wait_s=wait, attempt=attempt)
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    return None
                log.error("ss_api_error", status=resp.status_code, url=url)
                return None
            except requests.RequestException as exc:
                wait = _BACKOFF_BASE**attempt
                log.warning("request_failed", error=str(exc), wait_s=wait)
                time.sleep(wait)
        log.error("ss_max_retries_exceeded", url=url)
        return None

    def _to_paper(self, data: Dict) -> Optional[Paper]:
        """Convert a raw Semantic Scholar paper dict to a Paper dataclass."""
        external = data.get("externalIds") or {}
        arxiv_id = external.get("ArXiv")
        if not arxiv_id:
            return None

        year = data.get("year") or 2020
        authors = [a.get("name", "") for a in (data.get("authors") or [])]

        return Paper(
            id=arxiv_id,
            title=(data.get("title") or "").strip(),
            authors=authors,
            abstract="",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            published_date=datetime(year, 1, 1, tzinfo=timezone.utc),
            citation_count=data.get("citationCount") or 0,
            semantic_scholar_id=data.get("paperId"),
        )
