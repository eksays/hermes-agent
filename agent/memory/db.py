"""SQLite FTS5 storage layer for the Hermes file index.

Thread-safe, migration-gated store using WAL mode for concurrent
read/write performance and FTS5 for full-text search.
"""

import sqlite3
import threading
from typing import Any, Dict, List, Optional

_SCHEMA_VERSION = 3

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

-- v2: projects and git index

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    project_type TEXT DEFAULT 'unknown',
    framework TEXT DEFAULT '',
    build_tool TEXT DEFAULT '',
    last_active TEXT,
    indexed_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS project_deps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dep_name TEXT NOT NULL,
    dep_version TEXT DEFAULT '',
    is_dev INTEGER DEFAULT 0,
    dep_type TEXT DEFAULT 'npm'
);

CREATE TABLE IF NOT EXISTS git_repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    git_path TEXT NOT NULL,
    default_branch TEXT DEFAULT 'main',
    remote_url TEXT DEFAULT '',
    last_commit_hash TEXT DEFAULT '',
    last_commit_date TEXT,
    last_commit_message TEXT DEFAULT '',
    commit_count INTEGER DEFAULT 0,
    branch_count INTEGER DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS projects_fts USING fts5(
    name, root_path, framework,
    content='projects',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS projects_ai AFTER INSERT ON projects BEGIN
    INSERT INTO projects_fts(rowid, name, root_path, framework)
    VALUES (new.id, new.name, new.root_path, new.framework);
END;

CREATE TRIGGER IF NOT EXISTS projects_ad AFTER DELETE ON projects BEGIN
    INSERT INTO projects_fts(projects_fts, rowid, name, root_path, framework)
    VALUES ('delete', old.id, old.name, old.root_path, old.framework);
END;

CREATE TRIGGER IF NOT EXISTS projects_au AFTER UPDATE ON projects BEGIN
    INSERT INTO projects_fts(projects_fts, rowid, name, root_path, framework)
    VALUES ('delete', old.id, old.name, old.root_path, old.framework);
    INSERT INTO projects_fts(rowid, name, root_path, framework)
    VALUES (new.id, new.name, new.root_path, new.framework);
END;

-- v3: document content index

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    word_count INTEGER DEFAULT 0,
    summary TEXT DEFAULT '',
    indexed_at TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    content,
    content='documents',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, content)
    VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO documents_fts(rowid, content)
    VALUES (new.id, new.content);
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
                # v1→v2 migration: add projects, project_deps, git_repos tables
                if version < 2:
                    self._conn.executescript("""
                        CREATE TABLE IF NOT EXISTS projects (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            root_path TEXT UNIQUE NOT NULL,
                            name TEXT NOT NULL,
                            project_type TEXT DEFAULT 'unknown',
                            framework TEXT DEFAULT '',
                            build_tool TEXT DEFAULT '',
                            last_active TEXT,
                            indexed_at TEXT DEFAULT (datetime('now'))
                        );
                        CREATE TABLE IF NOT EXISTS project_deps (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                            dep_name TEXT NOT NULL,
                            dep_version TEXT DEFAULT '',
                            is_dev INTEGER DEFAULT 0,
                            dep_type TEXT DEFAULT 'npm'
                        );
                        CREATE TABLE IF NOT EXISTS git_repos (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                            git_path TEXT NOT NULL,
                            default_branch TEXT DEFAULT 'main',
                            remote_url TEXT DEFAULT '',
                            last_commit_hash TEXT DEFAULT '',
                            last_commit_date TEXT,
                            last_commit_message TEXT DEFAULT '',
                            commit_count INTEGER DEFAULT 0,
                            branch_count INTEGER DEFAULT 0
                        );
                        CREATE VIRTUAL TABLE IF NOT EXISTS projects_fts USING fts5(
                            name, root_path, framework,
                            content='projects', content_rowid='id',
                            tokenize='porter unicode61'
                        );
                        CREATE TRIGGER IF NOT EXISTS projects_ai AFTER INSERT ON projects BEGIN
                            INSERT INTO projects_fts(rowid, name, root_path, framework)
                            VALUES (new.id, new.name, new.root_path, new.framework); END;
                        CREATE TRIGGER IF NOT EXISTS projects_ad AFTER DELETE ON projects BEGIN
                            INSERT INTO projects_fts(projects_fts, rowid, name, root_path, framework)
                            VALUES ('delete', old.id, old.name, old.root_path, old.framework); END;
                        CREATE TRIGGER IF NOT EXISTS projects_au AFTER UPDATE ON projects BEGIN
                            INSERT INTO projects_fts(projects_fts, rowid, name, root_path, framework)
                            VALUES ('delete', old.id, old.name, old.root_path, old.framework);
                            INSERT INTO projects_fts(rowid, name, root_path, framework)
                            VALUES (new.id, new.name, new.root_path, new.framework); END;
                    """)
                # v2→v3: document content index
                if version < 3:
                    self._conn.executescript("""
                        CREATE TABLE IF NOT EXISTS documents (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            path TEXT UNIQUE NOT NULL,
                            content TEXT NOT NULL DEFAULT '',
                            content_hash TEXT NOT NULL,
                            word_count INTEGER DEFAULT 0,
                            summary TEXT DEFAULT '',
                            indexed_at TEXT DEFAULT (datetime('now'))
                        );
                        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                            content,
                            content='documents', content_rowid='id',
                            tokenize='porter unicode61'
                        );
                        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                            INSERT INTO documents_fts(rowid, content)
                            VALUES (new.id, new.content); END;
                        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                            INSERT INTO documents_fts(documents_fts, rowid, content)
                            VALUES ('delete', old.id, old.content); END;
                        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                            INSERT INTO documents_fts(documents_fts, rowid, content)
                            VALUES ('delete', old.id, old.content);
                            INSERT INTO documents_fts(rowid, content)
                            VALUES (new.id, new.content); END;
                    """)
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

    # ── Projects ─────────────────────────────────────────────────────────

    def upsert_project(
        self,
        root_path: str,
        name: str,
        project_type: str,
        framework: str = "",
        build_tool: str = "",
    ) -> int:
        """Insert or update a project record. Returns row id."""
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO projects (root_path, name, project_type, framework,
                                     build_tool, last_active, indexed_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(root_path) DO UPDATE SET
                    name         = excluded.name,
                    project_type = excluded.project_type,
                    framework    = excluded.framework,
                    build_tool   = excluded.build_tool,
                    last_active  = datetime('now'),
                    indexed_at   = datetime('now')
                """,
                (root_path, name, project_type, framework, build_tool),
            )
            return cursor.lastrowid

    def get_project_by_path(self, root_path: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM projects WHERE root_path = ?", (root_path,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def remove_project(self, root_path: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM projects WHERE root_path = ?", (root_path,))

    def clear_project_deps(self, project_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM project_deps WHERE project_id = ?", (project_id,)
            )

    def add_project_dep(self, project_id: int, name: str,
                        version: str, is_dev: bool, dep_type: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO project_deps (project_id, dep_name, dep_version,
                   is_dev, dep_type) VALUES (?, ?, ?, ?, ?)""",
                (project_id, name, version, 1 if is_dev else 0, dep_type),
            )
            return cursor.lastrowid

    def upsert_git_repo(
        self,
        project_id: int,
        git_path: str,
        default_branch: str = "main",
        remote_url: str = "",
        last_commit_hash: str = "",
        last_commit_date: str = "",
        last_commit_message: str = "",
        commit_count: int = 0,
        branch_count: int = 0,
    ) -> int:
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM git_repos WHERE project_id = ?", (project_id,)
            ).fetchone()
            if existing:
                self._conn.execute(
                    """UPDATE git_repos SET git_path=?, default_branch=?,
                       remote_url=?, last_commit_hash=?, last_commit_date=?,
                       last_commit_message=?, commit_count=?, branch_count=?
                     WHERE project_id=?""",
                    (git_path, default_branch, remote_url, last_commit_hash,
                     last_commit_date, last_commit_message, commit_count,
                     branch_count, project_id),
                )
                return existing[0]
            cursor = self._conn.execute(
                """INSERT INTO git_repos(project_id, git_path, default_branch,
                   remote_url, last_commit_hash, last_commit_date,
                   last_commit_message, commit_count, branch_count)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (project_id, git_path, default_branch, remote_url,
                 last_commit_hash, last_commit_date, last_commit_message,
                 commit_count, branch_count),
            )
            return cursor.lastrowid

    def search_projects(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        with self._lock:
            cursor = self._conn.execute(
                """SELECT p.* FROM projects p
                   JOIN projects_fts ON p.id = projects_fts.rowid
                   WHERE projects_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_project_paths(self) -> List[str]:
        with self._lock:
            cursor = self._conn.execute("SELECT root_path FROM projects")
            return [row["root_path"] for row in cursor.fetchall()]

    # ── Documents ────────────────────────────────────────────────────────

    def upsert_document(
        self,
        path: str,
        content: str,
        content_hash: str,
        word_count: int,
        summary: str = "",
    ) -> int:
        """Insert or update a document record. Returns row id."""
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO documents (path, content, content_hash, word_count, summary)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content      = excluded.content,
                    content_hash = excluded.content_hash,
                    word_count   = excluded.word_count,
                    summary      = excluded.summary,
                    indexed_at   = datetime('now')
                """,
                (path, content, content_hash, word_count, summary),
            )
            return cursor.lastrowid

    def get_document_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """Return the document record for *path*, or ``None``."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM documents WHERE path = ?", (path,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def search_documents(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Full-text search over document content via FTS5."""
        if not query.strip():
            return []
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT d.*
                FROM documents d
                JOIN documents_fts ON d.id = documents_fts.rowid
                WHERE documents_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def remove_document(self, path: str) -> None:
        """Delete the document at *path*."""
        with self._lock:
            self._conn.execute("DELETE FROM documents WHERE path = ?", (path,))

    def get_all_document_paths(self) -> List[str]:
        """Return all indexed document paths."""
        with self._lock:
            cursor = self._conn.execute("SELECT path FROM documents ORDER BY path")
            return [row["path"] for row in cursor.fetchall()]

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
            stats = dict(cursor.fetchone())
            stats["total_projects"] = self._conn.execute(
                "SELECT COUNT(*) FROM projects"
            ).fetchone()[0]
            stats["total_git_repos"] = self._conn.execute(
                "SELECT COUNT(*) FROM git_repos"
            ).fetchone()[0]
            stats["total_documents"] = self._conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            return stats

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
