"""SQLite FTS5 storage layer for the Hermes file index.

Thread-safe, migration-gated store using WAL mode for concurrent
read/write performance and FTS5 for full-text search.
"""

import sqlite3
import threading
from typing import Any, Dict, List, Optional

_SCHEMA_VERSION = 1

_INIT_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    extension TEXT DEFAULT '',
    size_bytes INTEGER DEFAULT 0,
    modified_at TEXT,
    file_type TEXT DEFAULT 'other',
    checksum TEXT DEFAULT '',
    is_dir INTEGER DEFAULT 0,
    indexed_at TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    path, filename,
    content='files',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, path, filename)
    VALUES (new.id, new.path, new.filename);
END;

CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, path, filename)
    VALUES ('delete', old.id, old.path, old.filename);
END;

CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, path, filename)
    VALUES ('delete', old.id, old.path, old.filename);
    INSERT INTO files_fts(rowid, path, filename)
    VALUES (new.id, new.path, new.filename);
END;
"""


class MemoryDB:
    """Thread-safe SQLite store for the Hermes file index.

    Creates and manages a ``files`` table and an FTS5 virtual table
    (``files_fts``) with automatic synchronisation triggers.

    Parameters
    ----------
    db_path : str
        Filesystem path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._lock = threading.Lock()

        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── Schema management ─────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Set up tables and triggers if this is a fresh database, or
        verify schema version on an existing one."""
        with self._lock:
            cursor = self._conn.execute("PRAGMA user_version")
            version = cursor.fetchone()[0]

            if version == 0:
                # Fresh database — execute init SQL and set schema version.
                self._conn.executescript(_INIT_SQL)
                self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version < _SCHEMA_VERSION:
                # Future migration point: run upgrade steps here.
                self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    # ── CRUD ──────────────────────────────────────────────────────────────

    def upsert_file(
        self,
        path: str,
        filename: str,
        extension: str,
        size_bytes: int,
        modified_at: str,
        file_type: str,
        checksum: str,
        is_dir: bool = False,
    ) -> int:
        """Insert a file record, or update an existing one by ``path``.

        Returns the row id of the inserted or updated record.
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO files (path, filename, extension, size_bytes,
                                   modified_at, file_type, checksum, is_dir)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    filename    = excluded.filename,
                    extension   = excluded.extension,
                    size_bytes  = excluded.size_bytes,
                    modified_at = excluded.modified_at,
                    file_type   = excluded.file_type,
                    checksum    = excluded.checksum,
                    is_dir      = excluded.is_dir,
                    indexed_at  = datetime('now')
                """,
                (
                    path,
                    filename,
                    extension,
                    size_bytes,
                    modified_at,
                    file_type,
                    checksum,
                    1 if is_dir else 0,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_file_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """Return the file record for *path*, or ``None`` if not found."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM files WHERE path = ?", (path,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def file_exists(self, path: str, checksum: str) -> bool:
        """Return ``True`` if a record with *path* and *checksum* exists."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT 1 FROM files WHERE path = ? AND checksum = ?",
                (path, checksum),
            )
            return cursor.fetchone() is not None

    def search_files(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-text search over path and filename via FTS5.

        Returns up to *limit* matching file records.
        """
        if not query.strip():
            return []
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT files.*
                FROM files
                JOIN files_fts ON files.id = files_fts.rowid
                WHERE files_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_file_paths(self) -> List[str]:
        """Return every indexed ``path`` in the store."""
        with self._lock:
            cursor = self._conn.execute("SELECT path FROM files ORDER BY path")
            return [row["path"] for row in cursor.fetchall()]

    def remove_file(self, path: str) -> None:
        """Delete the file record at *path*.

        The FTS ``files_ad`` trigger handles removing the FTS entry.
        """
        with self._lock:
            self._conn.execute("DELETE FROM files WHERE path = ?", (path,))

    # ── Introspection ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics about the indexed store."""
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT
                    COUNT(*)                           AS total_files,
                    COALESCE(SUM(size_bytes), 0)       AS total_size_bytes,
                    COALESCE(MAX(indexed_at), '')      AS total_indexed_at
                FROM files
                """
            )
            return dict(cursor.fetchone())

    def get_table_names(self) -> List[str]:
        """Return the list of user table names in the database."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            return [row["name"] for row in cursor.fetchall()]

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                finally:
                    self._conn = None  # type: ignore[assignment]
