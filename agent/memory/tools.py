"""Agent-facing tool facade for the Hermes Memory System.

Exposes :class:`MemoryManager` — a thin layer over
:class:`agent.memory.db.MemoryDB` that sanitises queries, applies
post-filtering, and provides OpenAI-compatible tool schemas for agent
function calling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.memory.config import DEFAULT_SEARCH_LIMIT
from agent.memory.db import MemoryDB
from agent.memory.safety import sanitize_query


class SearchResult:
    """Lightweight wrapper around a search result row.

    Parameters
    ----------
    All parameters match the ``files`` table columns returned by
    :meth:`MemoryDB.search_files`.
    """

    def __init__(
        self,
        id: int,
        path: str,
        filename: str,
        extension: str,
        size_bytes: int,
        modified_at: str,
        file_type: str,
        checksum: str,
        is_dir: int,
        indexed_at: str,
    ) -> None:
        self.id = id
        self.path = path
        self.filename = filename
        self.extension = extension
        self.size_bytes = size_bytes
        self.modified_at = modified_at
        self.file_type = file_type
        self.checksum = checksum
        self.is_dir = is_dir
        self.indexed_at = indexed_at

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary of all fields."""
        return {
            "id": self.id,
            "path": self.path,
            "filename": self.filename,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "file_type": self.file_type,
            "checksum": self.checksum,
            "is_dir": self.is_dir,
            "indexed_at": self.indexed_at,
        }


class MemoryManager:
    """Agent-facing facade over :class:`MemoryDB`.

    Provides search (with query sanitisation and post-filtering), status
    introspection, and OpenAI-compatible tool schemas for use with
    function-calling LLMs.

    Parameters
    ----------
    db : MemoryDB
        An initialised database handle.
    """

    def __init__(self, db: MemoryDB) -> None:
        self.db = db

    def search(
        self,
        query: str,
        type_filter: Optional[str] = None,
        ext_filter: Optional[str] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Search indexed files via FTS5 full-text search.

        The *query* is sanitised before being passed to the database
        (lowercased, stripped, special characters removed). Results can be
        further narrowed by *type_filter* (``file_type`` column) or
        *ext_filter* (``extension`` column).

        Parameters
        ----------
        query : str
            Free-text search string.
        type_filter : str or None
            If given, only return rows whose ``file_type`` matches this value.
        ext_filter : str or None
            If given, only return rows whose ``extension`` matches this value.
        limit : int
            Maximum number of results to return. Defaults to
            :const:`agent.memory.config.DEFAULT_SEARCH_LIMIT` (20).

        Returns
        -------
        list[dict]
            Each dict contains the columns of the ``files`` table.
        """
        sanitised = sanitize_query(query)
        if not sanitised:
            return []

        rows = self.db.search_files(sanitised, limit=limit)

        results: List[Dict[str, Any]] = []
        for row in rows:
            if type_filter is not None and row.get("file_type") != type_filter:
                continue
            if ext_filter is not None and row.get("extension") != ext_filter:
                continue
            results.append(row)

        return results[:limit]

    def status(self) -> Dict[str, Any]:
        """Return database statistics.

        Delegates to :meth:`MemoryDB.get_stats`.
        """
        return self.db.get_stats()

    # ── OpenAI-compatible tool schemas ──────────────────────────────────────

    @staticmethod
    def search_tool_schema() -> dict:
        """OpenAI function-calling schema for ``memory_search``."""
        return {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": (
                    "Search the indexed file memory by path or filename. "
                    "Optional type and extension filters narrow results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for file path or name.",
                        },
                        "type_filter": {
                            "type": "string",
                            "description": (
                                "Optional file type to filter by "
                                "(e.g. 'code', 'doc', 'config')."
                            ),
                        },
                        "ext_filter": {
                            "type": "string",
                            "description": (
                                "Optional file extension to filter by "
                                "(e.g. '.py', '.md', '.yaml')."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return.",
                            "default": DEFAULT_SEARCH_LIMIT,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    @staticmethod
    def index_tool_schema() -> dict:
        """OpenAI function-calling schema for ``memory_index``."""
        return {
            "type": "function",
            "function": {
                "name": "memory_index",
                "description": (
                    "Trigger an immediate re-scan of the configured roots "
                    "to refresh the memory index."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    @staticmethod
    def status_tool_schema() -> dict:
        """OpenAI function-calling schema for ``memory_status``."""
        return {
            "type": "function",
            "function": {
                "name": "memory_status",
                "description": (
                    "Return statistics about the indexed memory store "
                    "(total files, total size, last index time)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }
