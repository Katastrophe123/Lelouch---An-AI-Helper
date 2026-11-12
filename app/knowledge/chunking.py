"""Split text into overlapping passages for embedding."""

from __future__ import annotations

import re

_PARAGRAPH = re.compile(r"\n\s*\n")


def _hard_split(text: str, size: int) -> list[str]:
    """Split one over-long paragraph on whitespace near ``size``."""
    pieces: list[str] = []
    while len(text) > size:
        cut = text.rfind(" ", 0, size)
        cut = cut if cut > size // 2 else size
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def chunk_text(text: str, *, size: int = 1200, overlap: int = 200) -> list[str]:
    """Paragraph-aware chunks of at most ``size`` characters.

    Paragraphs are packed together until the next one would overflow. Each new
    chunk starts with the last ``overlap`` characters of the previous one, so a
    fact that straddles a boundary is still retrievable from either side.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    overlap = max(0, min(overlap, size // 2))
    paragraphs: list[str] = []
    for raw in _PARAGRAPH.split(text):
        paragraph = " ".join(raw.split())
        if paragraph:
            paragraphs.extend(_hard_split(paragraph, size - overlap))

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        if tail:
            space = tail.find(" ")
            tail = tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail
        current = f"{tail} {paragraph}".strip() if tail else paragraph
    if current:
        chunks.append(current)
    return chunks
