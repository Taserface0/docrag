"""PDF loading and chunking.

Chunks keep their source file name and page number so every answer can cite
exactly where it came from.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class Page:
    doc_id: str
    source: str
    page: int  # 1-based
    text: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    page: int
    index: int  # position within the document
    text: str

    def metadata(self) -> dict:
        return {"doc_id": self.doc_id, "source": self.source, "page": self.page, "index": self.index}


def doc_id_for(data: bytes) -> str:
    """Content hash, so uploading the same PDF twice is a no-op."""
    return hashlib.sha256(data).hexdigest()[:16]


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(data: bytes, source: str) -> list[Page]:
    reader = PdfReader(io.BytesIO(data))
    did = doc_id_for(data)
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        text = clean_text(p.extract_text() or "")
        if text:
            pages.append(Page(doc_id=did, source=source, page=i, text=text))
    return pages


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    """Recursively split on the coarsest separator that yields pieces <= size."""
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    out: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) > size:
            out.extend(_split(piece, size, rest))
        elif piece:
            out.append(piece)
    return out


def split_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Pack small pieces into chunks of ~size chars, carrying `overlap` chars forward."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    pieces = _split(text, size - overlap, _SEPARATORS)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # start the overlap at a word boundary
            if " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = tail
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_pages(pages: list[Page], size: int = 1000, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for page in pages:
        for text in split_text(page.text, size, overlap):
            if len(text) < 20:  # drop page-number-only fragments
                continue
            cid = f"{page.doc_id}-{idx:05d}"
            chunks.append(Chunk(cid, page.doc_id, page.source, page.page, idx, text))
            idx += 1
    return chunks
