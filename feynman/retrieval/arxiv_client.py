"""ArXiv API wrapper — fetches and filters recent research papers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import arxiv
import structlog

log = structlog.get_logger()


@dataclass
class Paper:
    """A research paper from ArXiv, optionally enriched by Semantic Scholar."""

    id: str
    title: str
    authors: List[str]
    abstract: str
    pdf_url: str
    published_date: datetime
    citation_count: int = 0
    semantic_scholar_id: Optional[str] = None


class ArXivClient:
    """Fetches papers from the ArXiv API sorted by relevance."""

    def __init__(self, years_back: int = 5) -> None:
        """
        Args:
            years_back: Exclude papers published more than this many years ago.
        """
        self.years_back = years_back
        # page_size=20 avoids requesting 100 results at once, which reliably
        # triggers ArXiv's 429 rate limit on the first cold request.
        # delay_seconds=5 and num_retries=5 give the server time to recover
        # when the limit is briefly hit anyway.
        self._client = arxiv.Client(
            page_size=20,
            delay_seconds=5,
            num_retries=5,
        )

    def search(self, query: str, max_results: int = 20) -> List[Paper]:
        """Search ArXiv by relevance and return recent papers.

        Fetches up to 2× max_results from the API, then filters by publish
        date so that the final list is always within the configured time window.

        Args:
            query: Natural language or boolean ArXiv search string.
            max_results: Maximum number of Paper objects to return.

        Returns:
            List[Paper] sorted by ArXiv relevance score (descending).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=365 * self.years_back)

        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: List[Paper] = []
        for result in self._client.results(search):
            pub = result.published
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub < cutoff:
                continue

            try:
                pdf_url = result.pdf_url or ""
            except Exception:
                pdf_url = f"https://arxiv.org/pdf/{result.entry_id.split('/abs/')[-1]}"

            papers.append(
                Paper(
                    id=result.entry_id.split("/abs/")[-1],
                    title=result.title.strip(),
                    authors=[a.name for a in result.authors],
                    abstract=result.summary.strip(),
                    pdf_url=pdf_url,
                    published_date=pub,
                    citation_count=0,
                )
            )
            if len(papers) >= max_results:
                break

        log.info("arxiv_search_done", query=query, returned=len(papers))
        return papers
