"""Extract text content from documents for the Hermes Memory System.

Supported formats: TXT, MD, PDF, DOCX — all handled offline via
stdlib or lightweight Python libraries already present in the venv.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Maximum characters allowed per document.
MAX_CHAR_LIMIT = 100_000

# Extensions this module can handle.
SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".pdf", ".docx"})


def extract_text(filepath: str) -> Optional[dict[str, Any]]:
    """Extract text content from a document.

    Parameters
    ----------
    filepath : str
        Absolute path to the document.

    Returns
    -------
    dict or None
        A dictionary with keys ``text`` (str), ``word_count`` (int),
        and ``extension`` (str), or ``None`` if the format is unsupported
        or the file cannot be read.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return None

    if not os.path.isfile(filepath):
        return None

    try:
        if ext in (".txt", ".md"):
            text = _read_text(filepath)
        elif ext == ".pdf":
            text = _read_pdf(filepath)
        elif ext == ".docx":
            text = _read_docx(filepath)
        else:
            return None
    except Exception:
        logger.exception("Failed to extract text from %s", filepath)
        return None

    if len(text) > MAX_CHAR_LIMIT:
        text = text[:MAX_CHAR_LIMIT]

    word_count = len(text.split()) if text.strip() else 0

    return {
        "text": text,
        "word_count": word_count,
        "extension": ext,
    }


# ── Format-specific readers ─────────────────────────────────────────────────


def _read_text(filepath: str) -> str:
    """Read a plain text (or markdown) file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Fall back to latin-1 if utf-8 fails (binary content).
        with open(filepath, "r", encoding="latin-1") as f:
            return f.read()


def _read_pdf(filepath: str) -> str:
    """Extract text from a PDF using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _read_docx(filepath: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    from docx import Document

    doc = Document(filepath)
    paragraphs: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)
