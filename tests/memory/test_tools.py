"""Tests for agent.memory.tools — MemoryManager and SearchResult."""

import os
import tempfile

from agent.memory.db import MemoryDB
from agent.memory.tools import MemoryManager, SearchResult


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_db() -> tuple[str, MemoryDB]:
    """Create a temp database and return (path, db) pair."""
    path = tempfile.mktemp(suffix=".db", prefix="hermes_test_")
    db = MemoryDB(path)
    return path, db


def _cleanup(path: str, db: MemoryDB) -> None:
    """Close the database and remove the file."""
    db.close()
    # SQLite WAL / SHM files on Windows may need a moment to release.
    # Retry once if needed.
    for db_file in (path, path + "-wal", path + "-shm"):
        if os.path.exists(db_file):
            try:
                os.unlink(db_file)
            except PermissionError:
                pass


# ── SearchResult ─────────────────────────────────────────────────────────────


def test_search_result_to_dict():
    """SearchResult.to_dict() returns the expected dictionary."""
    result = SearchResult(
        id=1,
        path="/project/main.py",
        filename="main.py",
        extension=".py",
        size_bytes=1024,
        modified_at="2025-01-15T10:00:00",
        file_type="code",
        checksum="abc123",
        is_dir=0,
        indexed_at="2025-01-15T10:00:00",
    )
    d = result.to_dict()
    assert d["path"] == "/project/main.py"
    assert d["filename"] == "main.py"
    assert d["extension"] == ".py"
    assert d["size_bytes"] == 1024
    assert d["is_dir"] == 0
    assert d["id"] == 1


# ── MemoryManager.search ─────────────────────────────────────────────────────


def test_memory_search_returns_results():
    """Upsert a file, search it, verify result via MemoryManager."""
    path, db = _make_db()
    try:
        mgr = MemoryManager(db)

        db.upsert_file(
            path="/project/src/main.py",
            filename="main.py",
            extension=".py",
            size_bytes=1024,
            modified_at="2025-01-15T10:00:00",
            file_type="code",
            checksum="abc123",
        )

        results = mgr.search("main")
        assert len(results) >= 1
        assert results[0]["path"] == "/project/src/main.py"
        assert results[0]["filename"] == "main.py"
    finally:
        _cleanup(path, db)


def test_memory_search_type_filter():
    """type_filter narrows results to matching file_type."""
    path, db = _make_db()
    try:
        mgr = MemoryManager(db)

        db.upsert_file(
            path="/project/src/main.py",
            filename="main.py",
            extension=".py",
            size_bytes=1024,
            modified_at="2025-01-15T10:00:00",
            file_type="code",
            checksum="a1",
        )
        db.upsert_file(
            path="/project/docs/readme.md",
            filename="readme.md",
            extension=".md",
            size_bytes=500,
            modified_at="2025-01-15T10:00:00",
            file_type="doc",
            checksum="a2",
        )

        # Search for 'main' but filter by 'doc' -- should find nothing
        results_doc = mgr.search("main", type_filter="doc")
        assert len(results_doc) == 0

        # Search for 'readme' with doc filter -- should find it
        results_readme = mgr.search("readme", type_filter="doc")
        assert len(results_readme) >= 1
    finally:
        _cleanup(path, db)


def test_memory_search_ext_filter():
    """ext_filter narrows results to matching extension."""
    path, db = _make_db()
    try:
        mgr = MemoryManager(db)

        db.upsert_file(
            path="/project/src/main.py",
            filename="main.py",
            extension=".py",
            size_bytes=1024,
            modified_at="2025-01-15T10:00:00",
            file_type="code",
            checksum="a1",
        )
        db.upsert_file(
            path="/project/docs/readme.md",
            filename="readme.md",
            extension=".md",
            size_bytes=500,
            modified_at="2025-01-15T10:00:00",
            file_type="doc",
            checksum="a2",
        )

        results_py = mgr.search("main", ext_filter=".py")
        assert len(results_py) >= 1

        results_md = mgr.search("main", ext_filter=".md")
        assert len(results_md) == 0
    finally:
        _cleanup(path, db)


def test_memory_search_empty_query():
    """Empty/whitespace query returns []."""
    path, db = _make_db()
    try:
        mgr = MemoryManager(db)

        db.upsert_file(
            path="/project/src/main.py",
            filename="main.py",
            extension=".py",
            size_bytes=1024,
            modified_at="2025-01-15T10:00:00",
            file_type="code",
            checksum="abc123",
        )

        assert mgr.search("") == []
        assert mgr.search("   ") == []
    finally:
        _cleanup(path, db)


def test_memory_search_limit():
    """limit parameter caps the number of results."""
    path, db = _make_db()
    try:
        mgr = MemoryManager(db)

        for i in range(10):
            db.upsert_file(
                path=f"/project/file_{i}.py",
                filename=f"file_{i}.py",
                extension=".py",
                size_bytes=100,
                modified_at="2025-01-15T10:00:00",
                file_type="code",
                checksum=f"c{i}",
            )

        results = mgr.search("file", limit=3)
        assert len(results) <= 3
    finally:
        _cleanup(path, db)


# ── MemoryManager.status ─────────────────────────────────────────────────────


def test_memory_status_returns_stats():
    """status returns dict that includes total_files."""
    path, db = _make_db()
    try:
        mgr = MemoryManager(db)

        db.upsert_file(
            path="/project/src/main.py",
            filename="main.py",
            extension=".py",
            size_bytes=1024,
            modified_at="2025-01-15T10:00:00",
            file_type="code",
            checksum="abc123",
        )

        stats = mgr.status()
        assert "total_files" in stats
        assert stats["total_files"] >= 1
    finally:
        _cleanup(path, db)


def test_memory_status_empty():
    """status on empty database returns total_files = 0."""
    path, db = _make_db()
    try:
        mgr = MemoryManager(db)

        stats = mgr.status()
        assert stats["total_files"] == 0
        assert stats["total_size_bytes"] == 0
    finally:
        _cleanup(path, db)


# ── Tool schemas ─────────────────────────────────────────────────────────────


def test_search_tool_schema():
    """search_tool_schema returns OpenAI function-calling format with query param."""
    schema = MemoryManager.search_tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "memory_search"
    assert "query" in schema["function"]["parameters"]["properties"]
    assert "query" in schema["function"]["parameters"]["required"]


def test_index_tool_schema():
    """index_tool_schema returns OpenAI function-calling format for memory_index."""
    schema = MemoryManager.index_tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "memory_index"


def test_status_tool_schema():
    """status_tool_schema returns OpenAI function-calling format for memory_status."""
    schema = MemoryManager.status_tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "memory_status"


# ── Document search scope ─────────────────────────────────────────────────────


def test_memory_search_document_scope():
    """scope='document' searches document content via FTS5."""
    path, db = _make_db()
    try:
        mgr = MemoryManager(db)
        db.upsert_document("/docs/botany.md", "Plants use photosynthesis.", "hash1", 3, "")
        db.upsert_document("/docs/astro.md", "Stars use nuclear fusion.", "hash2", 4, "")

        results = mgr.search("photosynthesis", scope="document")
        assert len(results) >= 1
        assert results[0]["_type"] == "document"

        results = mgr.search("fusion", scope="document")
        assert len(results) >= 1

        results = mgr.search("photosynthesis")
        assert len(results) >= 1  # default scope includes documents
    finally:
        _cleanup(path, db)
