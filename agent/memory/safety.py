"""Path validation, query sanitization, and PII filtering for the Hermes Memory System.

Coordinates
-----------
* `is_path_excluded` — basename-based exclude-pattern matching
* `sanitize_query` — lower-case / strip / special-char removal
* `strip_pii` — redact email, phone, and credit-card patterns from text
* `contains_pii` — boolean check for PII presence
"""

from __future__ import annotations

import re

# Characters to strip from search queries (keeps alphanumeric, spaces, dots, dashes, underscores, @, /, +, #).
_QUERY_CLEAN_RE = re.compile(r"[^\w\s.@/\-+#]")

# ── PII regex patterns ─────────────────────────────────────────────────────
# Email: standard RFC 5322 simplified.
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}"
    r"[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*"
)

# Phone: Indonesian (0812-, +62-) and general international patterns.
_PHONE_RE = re.compile(
    r"(?:\+?62[\s-]?|0)\d{1,4}[\s-]?\d{3,4}[\s-]?\d{3,8}"
)

# Credit card: 13-19 digit numbers in common formats (plain, spaced, dashed).
_CC_RE = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b"
)

# Combined for contains_pii fast scan.
_PII_ANY = re.compile(
    r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9]|"
    r"\+?62[\s-]?\d|\b\d{4}[-\s]?\d{4}|\b\d{16}\b"
)

_REDACTED = "[REDACTED]"


def is_path_excluded(path: str, exclude_patterns: set) -> bool:
    """Check if any path segment matches an exclude pattern (basename match)."""
    parts = path.replace("\\", "/").split("/")
    for pattern in exclude_patterns:
        if pattern in parts:
            return True
    return False


def sanitize_query(query: str) -> str:
    """Sanitize a search query: lowercase, strip, remove special characters."""
    if not query or not query.strip():
        return ""
    cleaned = _QUERY_CLEAN_RE.sub(" ", query)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip().lower()


def strip_pii(text: str) -> str:
    """Remove PII (email, phone, credit-card numbers) from *text*.

    Each detected PII segment is replaced with ``[REDACTED]`` so the
    caller still knows *something* was removed.

    Returns the redacted string.
    """
    if not text or not text.strip():
        return text

    result = _CC_RE.sub(_REDACTED, text)
    result = _EMAIL_RE.sub(_REDACTED, result)
    result = _PHONE_RE.sub(_REDACTED, result)
    return result


def contains_pii(text: str) -> bool:
    """Return ``True`` if *text* contains any PII pattern."""
    if not text:
        return False
    return bool(_PII_ANY.search(text))
