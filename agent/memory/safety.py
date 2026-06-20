"""Path validation and query sanitization for the Hermes Memory System."""

import re

_QUERY_CLEAN_RE = re.compile(r"[^\w\s.@/\-+#]")


def is_path_excluded(path: str, exclude_patterns: set) -> bool:
    """Check if any path segment matches an exclude pattern."""
    parts = path.replace("\\", "/").split("/")
    for pattern in exclude_patterns:
        if pattern in parts:
            return True
    return False


def sanitize_query(query: str) -> str:
    """Sanitize a search query: lowercase, strip, remove special chars."""
    if not query or not query.strip():
        return ""
    cleaned = _QUERY_CLEAN_RE.sub(" ", query)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip().lower()
