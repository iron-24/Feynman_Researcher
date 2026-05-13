"""PDF download, text extraction, section splitting, and chunking."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import requests
import structlog
import tiktoken

log = structlog.get_logger()

_SECTION_RE = re.compile(
    r"^(?:"
    r"abstract"
    r"|introduction"
    r"|related\s+work"
    r"|background"
    r"|method(?:ology|s)?"
    r"|approach"
    r"|model"
    r"|experiment(?:s|al\s+(?:setup|results?))?"
    r"|results?"
    r"|evaluation"
    r"|discussion"
    r"|conclusion(?:s)?"
    r"|references"
    r"|\d+\.?\s+\w[\w\s]{2,40}"  # numbered sections like "3. Proposed Method"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_FIGURE_RE = re.compile(
    r"(?:Figure|Fig\.|Table)\s+\d+[.:]?\s+[^\n]+(?:\n[A-Z][^\n]+){0,3}",
    re.IGNORECASE,
)


@dataclass
class TextChunk:
    """A tokenized segment of a paper section, ready for embedding."""

    paper_id: str
    title: str
    section: str
    chunk_index: int
    text: str


class PDFParser:
    """Downloads and parses ArXiv PDFs into structured, embeddable chunks.

    Chunking uses tiktoken for accurate token counts so that each chunk
    fits within the 512-token context window of all-MiniLM-L6-v2.
    """

    CHUNK_TOKENS: int = 512
    OVERLAP_TOKENS: int = 50

    def __init__(self) -> None:
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, paper_id: str, title: str, pdf_url: str) -> List[TextChunk]:
        """Full pipeline: download → extract → split → clean → chunk.

        Args:
            paper_id: ArXiv ID used in chunk metadata.
            title: Paper title used in chunk metadata.
            pdf_url: Direct link to the ArXiv PDF.

        Returns:
            Ordered list of TextChunk objects, or [] on failure.
        """
        raw = self.extract_text(pdf_url)
        if not raw:
            return []

        sections = self.extract_sections(raw)
        chunks: List[TextChunk] = []
        idx = 0
        for section_name, section_text in sections.items():
            clean = self.clean_text(section_text)
            for text in self._split_into_chunks(clean):
                chunks.append(
                    TextChunk(
                        paper_id=paper_id,
                        title=title,
                        section=section_name,
                        chunk_index=idx,
                        text=text,
                    )
                )
                idx += 1

        log.info("pdf_parsed", paper_id=paper_id, sections=len(sections), chunks=len(chunks))
        return chunks

    def extract_text(self, pdf_url: str) -> Optional[str]:
        """Download a PDF and return its full raw text.

        Args:
            pdf_url: Direct URL to the PDF file.

        Returns:
            Extracted text string, or None if download/parse fails.
        """
        try:
            resp = requests.get(pdf_url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.error("pdf_download_failed", url=pdf_url, error=str(exc))
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp.write(resp.content)
            tmp.close()
            doc = fitz.open(tmp.name)
            pages = [page.get_text("text") for page in doc]
            doc.close()
            return "\n".join(pages)
        except Exception as exc:
            log.error("pdf_parse_failed", url=pdf_url, error=str(exc))
            return None
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Split paper text into named sections using header detection.

        Falls back to {"full_text": text} if no section headers are found,
        which happens on heavily formatted or two-column PDFs.

        Args:
            text: Full raw text from extract_text().

        Returns:
            Ordered dict of {section_name: section_text}.
        """
        matches = list(_SECTION_RE.finditer(text))
        if not matches:
            return {"full_text": text}

        sections: Dict[str, str] = {}
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections["preamble"] = preamble

        for i, match in enumerate(matches):
            name = match.group(0).strip().lower()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if content:
                sections[name] = content

        return sections

    def clean_text(self, text: str) -> str:
        """Remove PDF extraction noise: hyphenation, footers, excess whitespace.

        Args:
            text: Raw section text.

        Returns:
            Cleaned text.
        """
        text = re.sub(r"-\n(\w)", r"\1", text)          # fix soft hyphenation
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # lone page numbers
        text = re.sub(r"\n{3,}", "\n\n", text)           # collapse blank lines
        text = re.sub(r"[ \t]{2,}", " ", text)           # collapse spaces
        return text.strip()

    def extract_figure_captions(self, text: str) -> List[str]:
        """Extract figure and table captions (text only, no images).

        Args:
            text: Full or section text.

        Returns:
            List of caption strings.
        """
        return [m.group(0).strip() for m in _FIGURE_RE.finditer(text)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_chunks(self, text: str) -> List[str]:
        """Tokenize text and split into overlapping fixed-size windows.

        Each window is at most CHUNK_TOKENS tokens.  Consecutive windows
        share OVERLAP_TOKENS tokens so concepts that cross a boundary are
        represented in both neighbours.

        Args:
            text: Cleaned section text.

        Returns:
            List of decoded text strings.
        """
        tokens = self._tokenizer.encode(text)
        if not tokens:
            return []

        chunks: List[str] = []
        start = 0
        stride = self.CHUNK_TOKENS - self.OVERLAP_TOKENS
        while start < len(tokens):
            end = min(start + self.CHUNK_TOKENS, len(tokens))
            chunks.append(self._tokenizer.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += stride

        return chunks
