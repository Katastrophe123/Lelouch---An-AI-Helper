"""Extract plain text from uploaded files."""

from __future__ import annotations

import io
from pathlib import PurePath

TEXT_EXTENSIONS = frozenset(
    {
        ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml",
        ".html", ".htm", ".xml", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".go",
        ".rs", ".sql", ".tex", ".log",
    }
)  # fmt: skip
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx"}


class UnsupportedDocumentError(Exception):
    """The file type cannot be turned into text."""


class EmptyDocumentError(Exception):
    """The file was readable but contained no text (e.g. a scanned PDF)."""


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[page {number}]\n{text}")
    return "\n\n".join(pages)


def _docx_text(data: bytes) -> str:
    from docx import Document

    document = Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def extract_text(filename: str, data: bytes) -> str:
    """Return the document's text, or raise if it has none we can read."""
    suffix = PurePath(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported file type {suffix or '(none)'}; "
            f"supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    try:
        if suffix == ".pdf":
            text = _pdf_text(data)
        elif suffix == ".docx":
            text = _docx_text(data)
        else:
            text = _decode(data)
    except (UnsupportedDocumentError, EmptyDocumentError):
        raise
    except Exception as exc:  # noqa: BLE001 - any parser failure is the file's fault
        raise UnsupportedDocumentError(f"Could not read {filename}: {exc}") from exc
    text = text.replace("\x00", "").strip()
    if not text:
        raise EmptyDocumentError(
            f"{filename} contains no extractable text (a scanned PDF needs OCR first)"
        )
    return text
