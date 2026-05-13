"""FAISS vector store for paper chunks with sentence-transformer embeddings."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from feynman.retrieval.pdf_parser import TextChunk

log = structlog.get_logger()

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384


@dataclass
class SearchResult:
    """A retrieved chunk paired with its cosine similarity score."""

    chunk: TextChunk
    score: float


class VectorStore:
    """FAISS IndexFlatIP store for semantic search over paper chunks.

    Uses inner product on L2-normalised embeddings, which is equivalent to
    cosine similarity.  For the typical dataset size (≤ 15 papers × ~100
    chunks), exact search is fast enough that approximate indices (IVF/HNSW)
    add complexity without benefit.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        """
        Args:
            model_name: sentence-transformers model ID for encoding queries
                        and document chunks.  Must be the same model used
                        when the index was built.
        """
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = SentenceTransformer(model_name, device=device)
        self._index = faiss.IndexFlatIP(_EMBEDDING_DIM)
        self._chunks: List[TextChunk] = []
        log.info("vector_store_init", model=model_name, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_paper(self, chunks: List[TextChunk]) -> None:
        """Embed and index all chunks for a single paper.

        Args:
            chunks: Output of PDFParser.parse() for one paper.
        """
        if not chunks:
            return
        embeddings = self._embed([c.text for c in chunks])
        self._index.add(embeddings)
        self._chunks.extend(chunks)
        log.info(
            "paper_indexed",
            paper_id=chunks[0].paper_id,
            chunks=len(chunks),
            total=self._index.ntotal,
        )

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Return the top-k most relevant chunks for a natural language query.

        Args:
            query: Question or keyword string to search for.
            k: Number of results to return.

        Returns:
            List of SearchResult sorted by descending similarity score.
        """
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        vec = self._embed([query])
        scores, indices = self._index.search(vec, k)
        return [
            SearchResult(chunk=self._chunks[i], score=float(s))
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and chunk metadata to disk.

        Args:
            directory: Target directory (created if it doesn't exist).
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as fh:
            pickle.dump(self._chunks, fh)
        log.info("vector_store_saved", path=str(path), vectors=self._index.ntotal)

    @classmethod
    def load(cls, directory: str | Path, model_name: str = _MODEL_NAME) -> "VectorStore":
        """Load a previously saved VectorStore from disk.

        Args:
            directory: The directory written by save().
            model_name: Must match the model used when save() was called.

        Returns:
            A VectorStore ready for search().
        """
        path = Path(directory)
        store = cls(model_name=model_name)
        store._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as fh:
            store._chunks = pickle.load(fh)
        log.info(
            "vector_store_loaded",
            path=str(path),
            vectors=store._index.ntotal,
            chunks=len(store._chunks),
        )
        return store

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Encode texts to L2-normalised float32 embeddings."""
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
