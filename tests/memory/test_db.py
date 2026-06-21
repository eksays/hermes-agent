"""Tests for agent.memory.db — SQLite FTS5 storage layer."""

import os
import sqlite3
import tempfile

import pytest

from agent.memory.db import MemoryDB


@pytest.fixture
def db_path() -> str:
    """Provide a temporary database path, cleaned up after the test."""
    tmp = tempfile.mktemp(suffix=".db", prefix="hermes_test_")
    yield tmp
    if os.path.exists(tmp):
        os.unlink(tmp)


# ── Init ────────────────────────────────────────────────────────────────────


def test_init_creates_tables(db_path):
    """MemoryDB creates tables and sets user_version = 5."""
    db = MemoryDB(db_path)
    try:
        tables = db.get_table_names()
        assert "files" in tables
        assert "files_fts" in tables
        assert "documents" in tables
        assert "documents_fts" in tables
        assert "memory_facts" in tables
        assert "memory_facts_fts" in tables
        assert "user_preferences" in tables
        # Verify v5 tables also exist
        assert "scored_items" in tables
        assert "user_boosts" in tables
        assert "access_log" in tables
        assert "doc_vectors" in tables
        assert "stale_config" in tables
        # Verify schema version via raw connection
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute("PRAGMA user_version")
            version = cur.fetchone()[0]
            assert version == 5
        finally:
            conn.close()
    finally:
        db.close()


def test_init_loads_existing_db(db_path):
    """Opening an existing database reuses tables and schema version."""
    db1 = MemoryDB(db_path)
    db1.close()
    db2 = MemoryDB(db_path)
    try:
        tables = db2.get_table_names()
        assert "files" in tables
        assert "files_fts" in tables
    finally:
        db2.close()


# ── Upsert (new) ────────────────────────────────────────────────────────────


def test_upsert_file_new(db_path):
    """Insert a new file returns its id and get_file_by_path returns data."""
    db = MemoryDB(db_path)
    try:
        row_id = db.upsert_file(
            path="/project/src/main.py",
            filename="main.py",
            extension=".py",
            size_bytes=1024,
            modified_at="2025-01-15T10:00:00",
            file_type="code",
            checksum="abc123",
            is_dir=False,
        )
        assert isinstance(row_id, int) and row_id > 0

        record = db.get_file_by_path("/project/src/main.py")
        assert record is not None
        assert record["path"] == "/project/src/main.py"
        assert record["filename"] == "main.py"
        assert record["extension"] == ".py"
        assert record["size_bytes"] == 1024
        assert record["modified_at"] == "2025-01-15T10:00:00"
        assert record["file_type"] == "code"
        assert record["checksum"] == "abc123"
        assert record["is_dir"] == 0
        assert "indexed_at" in record
        assert record["id"] == row_id
    finally:
        db.close()


# ── Upsert (update) ─────────────────────────────────────────────────────────


def test_upsert_file_updates_existing(db_path):
    """Re-inserting the same path updates the row and returns the same id."""
    db = MemoryDB(db_path)
    try:
        id1 = db.upsert_file(
            path="/project/README.md",
            filename="README.md",
            extension=".md",
            size_bytes=500,
            modified_at="2025-01-01T00:00:00",
            file_type="doc",
            checksum="old",
            is_dir=False,
        )
        id2 = db.upsert_file(
            path="/project/README.md",
            filename="README.md",
            extension=".md",
            size_bytes=800,
            modified_at="2025-06-01T00:00:00",
            file_type="doc",
            checksum="new",
            is_dir=False,
        )

        # Same row id (upsert)
        assert id1 == id2

        record = db.get_file_by_path("/project/README.md")
        assert record is not None
        assert record["size_bytes"] == 800
        assert record["modified_at"] == "2025-06-01T00:00:00"
        assert record["checksum"] == "new"
    finally:
        db.close()


# ── file_exists ─────────────────────────────────────────────────────────────


def test_file_exists(db_path):
    """file_exists returns True when path+checksum match, else False."""
    db = MemoryDB(db_path)
    try:
        db.upsert_file(
            path="/project/config.yaml",
            filename="config.yaml",
            extension=".yaml",
            size_bytes=200,
            modified_at="2025-03-01T00:00:00",
            file_type="config",
            checksum="xyz789",
            is_dir=False,
        )

        # Same path and checksum
        assert db.file_exists("/project/config.yaml", "xyz789") is True
        # Same path, different checksum
        assert db.file_exists("/project/config.yaml", "different") is False
        # Unknown path
        assert db.file_exists("/project/missing.yaml", "xyz789") is False
    finally:
        db.close()


# ── FTS search ──────────────────────────────────────────────────────────────


def test_search_files_fts(db_path):
    """FTS5 search returns matching results based on path and filename."""
    db = MemoryDB(db_path)
    try:
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
            path="/project/src/utils.py",
            filename="utils.py",
            extension=".py",
            size_bytes=2048,
            modified_at="2025-02-01T10:00:00",
            file_type="code",
            checksum="a2",
        )
        db.upsert_file(
            path="/project/docs/readme.md",
            filename="readme.md",
            extension=".md",
            size_bytes=500,
            modified_at="2025-03-01T10:00:00",
            file_type="doc",
            checksum="a3",
        )

        # Search by filename
        results = db.search_files("main", limit=20)
        assert len(results) >= 1
        assert any(r["path"] == "/project/src/main.py" for r in results)

        # Search by path segment
        results = db.search_files("utils", limit=20)
        assert len(results) >= 1
        assert any(r["path"] == "/project/src/utils.py" for r in results)

        # Search for something in docs
        results = db.search_files("readme", limit=20)
        assert len(results) >= 1
        assert any(r["path"] == "/project/docs/readme.md" for r in results)

        # No match
        results = db.search_files("zzzznothing", limit=20)
        assert results == []

        # limit works
        all_results = db.search_files("py", limit=20)
        limited = db.search_files("py", limit=1)
        assert len(limited) <= 1
        assert len(limited) < len(all_results)
    finally:
        db.close()


# ── get_all_file_paths ──────────────────────────────────────────────────────


def test_get_all_file_paths(db_path):
    """get_all_file_paths returns every path in the store."""
    db = MemoryDB(db_path)
    try:
        paths = ["/a.txt", "/b.txt", "/c.txt"]
        for i, p in enumerate(paths):
            db.upsert_file(
                path=p,
                filename=p.lstrip("/"),
                extension=".txt",
                size_bytes=100 + i,
                modified_at="2025-01-01T00:00:00",
                file_type="text",
                checksum=f"c{i}",
            )

        result = db.get_all_file_paths()
        assert sorted(result) == sorted(paths)
    finally:
        db.close()


# ── remove_file ─────────────────────────────────────────────────────────────


def test_remove_file(db_path):
    """remove_file deletes the file and it no longer appears."""
    db = MemoryDB(db_path)
    try:
        db.upsert_file(
            path="/project/to_delete.py",
            filename="to_delete.py",
            extension=".py",
            size_bytes=100,
            modified_at="2025-01-01T00:00:00",
            file_type="code",
            checksum="del",
        )
        assert db.get_file_by_path("/project/to_delete.py") is not None

        db.remove_file("/project/to_delete.py")
        assert db.get_file_by_path("/project/to_delete.py") is None

        # Removing a non-existent path is a no-op
        db.remove_file("/project/never_existed.py")  # should not raise
    finally:
        db.close()


# ── get_stats ───────────────────────────────────────────────────────────────


def test_get_stats_empty(db_path):
    """get_stats returns zero counts on an empty database."""
    db = MemoryDB(db_path)
    try:
        stats = db.get_stats()
        assert stats["total_files"] == 0
        assert stats["total_size_bytes"] == 0
    finally:
        db.close()


def test_get_stats(db_path):
    """get_stats returns correct total_files and total_size_bytes."""
    db = MemoryDB(db_path)
    try:
        db.upsert_file(
            path="/project/src/main.py",
            filename="main.py",
            extension=".py",
            size_bytes=1000,
            modified_at="2025-01-01T00:00:00",
            file_type="code",
            checksum="a",
        )
        db.upsert_file(
            path="/project/src/utils.py",
            filename="utils.py",
            extension=".py",
            size_bytes=2000,
            modified_at="2025-01-01T00:00:00",
            file_type="code",
            checksum="b",
        )
        db.upsert_file(
            path="/project/docs/readme.md",
            filename="readme.md",
            extension=".md",
            size_bytes=500,
            modified_at="2025-01-01T00:00:00",
            file_type="doc",
            checksum="c",
        )

        stats = db.get_stats()
        assert stats["total_files"] == 3
        assert stats["total_size_bytes"] == 3500
        assert "total_indexed_at" in stats
    finally:
        db.close()


# ── Concurrency guard ───────────────────────────────────────────────────────


def test_thread_safety(db_path):
    """Multiple operations on the same database should not crash."""
    import concurrent.futures

    db = MemoryDB(db_path)
    try:
        paths = [f"/project/file_{i}.py" for i in range(20)]

        def _upsert(p: str) -> int:
            return db.upsert_file(
                path=p,
                filename=p.split("/")[-1],
                extension=".py",
                size_bytes=100,
                modified_at="2025-01-01T00:00:00",
                file_type="code",
                checksum=f"c{p[-5]}",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(_upsert, paths))

        all_paths = db.get_all_file_paths()
        assert len(all_paths) == 20
    finally:
        db.close()


def test_close_idempotent(db_path):
    """Calling close multiple times is safe."""
    db = MemoryDB(db_path)
    db.close()
    db.close()  # second call should not raise


# ── Documents ─────────────────────────────────────────────────────────────────


def test_upsert_document_new(db_path):
    """Insert a new document record and retrieve it."""
    db = MemoryDB(db_path)
    try:
        doc_id = db.upsert_document(
            path="/docs/report.md",
            content="# Full markdown text content here for testing purposes",
            content_hash="abc123",
            word_count=8,
            summary="Short summary here.",
        )
        assert isinstance(doc_id, int) and doc_id > 0

        record = db.get_document_by_path("/docs/report.md")
        assert record is not None
        assert record["path"] == "/docs/report.md"
        assert record["content_hash"] == "abc123"
        assert record["word_count"] == 8
        assert "summary" in record
        assert "indexed_at" in record
    finally:
        db.close()


def test_upsert_document_updates_content(db_path):
    """Re-inserting the same path updates content and hash."""
    db = MemoryDB(db_path)
    try:
        db.upsert_document("/docs/paper.md", "v1 content", "hash1", 2, "")
        db.upsert_document("/docs/paper.md", "v2 content updated", "hash2", 3, "new summary")
        record = db.get_document_by_path("/docs/paper.md")
        assert record is not None
        assert record["content_hash"] == "hash2"
        assert record["word_count"] == 3
    finally:
        db.close()


def test_search_documents_fts(db_path):
    """FTS5 search returns matching documents by content."""
    db = MemoryDB(db_path)
    try:
        db.upsert_document("/docs/botany.txt", "Plants use photosynthesis to grow.", "a", 6, "")
        db.upsert_document("/docs/astronomy.txt", "Stars undergo nuclear fusion.", "b", 4, "")

        results = db.search_documents("photosynthesis", limit=20)
        assert len(results) >= 1
        assert any(r["path"] == "/docs/botany.txt" for r in results)

        results = db.search_documents("fusion", limit=20)
        assert len(results) >= 1
        assert any(r["path"] == "/docs/astronomy.txt" for r in results)

        results = db.search_documents("zzzznothing", limit=20)
        assert results == []
    finally:
        db.close()


def test_remove_document(db_path):
    """remove_document deletes a document record."""
    db = MemoryDB(db_path)
    try:
        db.upsert_document("/docs/temp.md", "content", "hash", 1, "")
        assert db.get_document_by_path("/docs/temp.md") is not None
        db.remove_document("/docs/temp.md")
        assert db.get_document_by_path("/docs/temp.md") is None
    finally:
        db.close()


def test_get_all_document_paths(db_path):
    """get_all_document_paths returns all indexed document paths."""
    db = MemoryDB(db_path)
    try:
        paths = ["/a.md", "/b.md", "/c.md"]
        for i, p in enumerate(paths):
            db.upsert_document(p, f"content {i}", f"h{i}", 2, "")
        result = db.get_all_document_paths()
        assert sorted(result) == sorted(paths)
    finally:
        db.close()


def test_get_stats_includes_documents(db_path):
    """get_stats returns total_documents count."""
    db = MemoryDB(db_path)
    try:
        db.upsert_document("/docs/a.md", "aaa", "h1", 1, "")
        db.upsert_document("/docs/b.md", "bbb", "h2", 1, "")
        stats = db.get_stats()
        assert stats["total_documents"] == 2
    finally:
        db.close()


# ── Memory Facts ──────────────────────────────────────────────────────────────


def test_upsert_memory_fact_new(db_path):
    """Insert a new memory fact and retrieve it by key."""
    db = MemoryDB(db_path)
    try:
        fact_id = db.upsert_memory_fact(
            key="user_preferred_editor",
            content="User likes VS Code with Python extension",
            category="core",
            source="user",
        )
        assert isinstance(fact_id, int) and fact_id > 0

        record = db.get_memory_fact("user_preferred_editor")
        assert record is not None
        assert record["content"] == "User likes VS Code with Python extension"
        assert record["category"] == "core"
        assert record["source"] == "user"
    finally:
        db.close()


def test_upsert_memory_fact_updates(db_path):
    """Re-inserting same key updates content and metadata."""
    db = MemoryDB(db_path)
    try:
        db.upsert_memory_fact("key1", "v1", "core", "user")
        db.upsert_memory_fact("key1", "v2", "daily", "system")
        record = db.get_memory_fact("key1")
        assert record is not None
        assert record["content"] == "v2"
        assert record["category"] == "daily"
    finally:
        db.close()


def test_memory_fact_not_found(db_path):
    """get_memory_fact returns None for missing key."""
    db = MemoryDB(db_path)
    try:
        assert db.get_memory_fact("nonexistent") is None
    finally:
        db.close()


def test_search_memory_facts_fts(db_path):
    """FTS5 search returns matching memory facts."""
    db = MemoryDB(db_path)
    try:
        db.upsert_memory_fact("pref_stack", "User prefers Python stack", "core", "user")
        db.upsert_memory_fact("pref_editor", "User prefers VS Code", "core", "user")

        results = db.search_memory_facts("Python")
        assert len(results) >= 1
        assert any(r["key"] == "pref_stack" for r in results)

        results = db.search_memory_facts("VS Code")
        assert len(results) >= 1

        results = db.search_memory_facts("zzzznothing")
        assert results == []
    finally:
        db.close()


def test_delete_memory_fact(db_path):
    """delete_memory_fact removes a fact by key."""
    db = MemoryDB(db_path)
    try:
        db.upsert_memory_fact("key_del", "to delete", "core", "user")
        assert db.get_memory_fact("key_del") is not None
        db.delete_memory_fact("key_del")
        assert db.get_memory_fact("key_del") is None
    finally:
        db.close()


def test_get_all_memory_fact_keys(db_path):
    """get_all_memory_fact_keys returns all stored keys."""
    db = MemoryDB(db_path)
    try:
        keys = ["k1", "k2", "k3"]
        for k in keys:
            db.upsert_memory_fact(k, f"content of {k}", "core", "user")
        result = db.get_all_memory_fact_keys()
        assert sorted(result) == sorted(keys)
    finally:
        db.close()


# ── User Preferences ──────────────────────────────────────────────────────────


def test_upsert_preference_new(db_path):
    """Insert a new user preference and retrieve it."""
    db = MemoryDB(db_path)
    try:
        pref_id = db.upsert_preference(
            key="communication_style",
            value="concise and direct",
            category="behavior",
        )
        assert isinstance(pref_id, int) and pref_id > 0

        record = db.get_preference("communication_style")
        assert record is not None
        assert record["value"] == "concise and direct"
        assert record["category"] == "behavior"
    finally:
        db.close()


def test_upsert_preference_updates(db_path):
    """Re-inserting same preference key updates value."""
    db = MemoryDB(db_path)
    try:
        db.upsert_preference("style", "verbose", "behavior")
        db.upsert_preference("style", "concise", "behavior")
        record = db.get_preference("style")
        assert record["value"] == "concise"
    finally:
        db.close()


def test_get_all_preference_keys(db_path):
    """get_all_preference_keys returns all stored preference keys."""
    db = MemoryDB(db_path)
    try:
        db.upsert_preference("p1", "v1", "behavior")
        db.upsert_preference("p2", "v2", "schedule")
        keys = db.get_all_preference_keys()
        assert "p1" in keys
        assert "p2" in keys
    finally:
        db.close()


def test_delete_preference(db_path):
    """delete_preference removes a preference by key."""
    db = MemoryDB(db_path)
    try:
        db.upsert_preference("del_me", "value", "behavior")
        assert db.get_preference("del_me") is not None
        db.delete_preference("del_me")
        assert db.get_preference("del_me") is None
    finally:
        db.close()


def test_get_stats_includes_memory(db_path):
    """get_stats returns memory_facts and preferences counts."""
    db = MemoryDB(db_path)
    try:
        db.upsert_memory_fact("f1", "fact 1", "core", "user")
        db.upsert_preference("p1", "pref 1", "behavior")

        stats = db.get_stats()
        assert stats["total_memory_facts"] >= 1
        assert stats["total_preferences"] >= 1
    finally:
        db.close()


# ── TTL expiry ────────────────────────────────────────────────────────────────


def test_cleanup_expired_facts_removes_expired(db_path):
    """cleanup_expired_facts removes facts past their TTL."""
    db = MemoryDB(db_path)
    try:
        db.upsert_memory_fact("keep_me", "permanent", "core", "user")
        db.upsert_memory_fact("expired", "old data", "daily", "system", ttl_seconds=1)
        import time
        time.sleep(1.1)
        removed = db.cleanup_expired_facts()
        assert removed >= 1
        assert db.get_memory_fact("keep_me") is not None
        assert db.get_memory_fact("expired") is None
    finally:
        db.close()


def test_cleanup_expired_facts_no_expiry(db_path):
    """Facts with ttl_seconds=None are never removed."""
    db = MemoryDB(db_path)
    try:
        db.upsert_memory_fact("core1", "data1", "core", "user")
        db.upsert_memory_fact("core2", "data2", "core", "user")
        removed = db.cleanup_expired_facts()
        assert removed == 0
        assert db.get_memory_fact("core1") is not None
    finally:
        db.close()


def test_cleanup_expired_facts_empty(db_path):
    """cleanup_expired_facts on empty DB returns 0."""
    db = MemoryDB(db_path)
    try:
        assert db.cleanup_expired_facts() == 0
    finally:
        db.close()



# ── Preferences standalone ─────────────────────────────────────────────────────


def test_get_all_preferences_returns_list(db_path):
    """get_all_preferences returns all preference records."""
    db = MemoryDB(db_path)
    try:
        db.upsert_preference("p1", "v1", "behavior")
        db.upsert_preference("p2", "v2", "schedule")
        all_prefs = db.get_all_preferences()
        assert len(all_prefs) >= 2
        keys = [p["key"] for p in all_prefs]
        assert "p1" in keys
        assert "p2" in keys
    finally:
        db.close()


# ── v5: Scoring tables ──────────────────────────────────────────────────────


def test_init_creates_v5_tables(db_path):
    """v5 init creates scored_items, user_boosts, access_log, doc_vectors."""
    db = MemoryDB(db_path)
    try:
        tables = db.get_table_names()
        assert "scored_items" in tables
        assert "user_boosts" in tables
        assert "access_log" in tables
        assert "doc_vectors" in tables
        assert db.get_user_version() == 5
    finally:
        db.close()


def test_set_user_boost_creates(db_path):
    """set_user_boost inserts a boost record."""
    db = MemoryDB(db_path)
    try:
        db.set_user_boost("file", 1, "important", 2.5)
        boosts = db.get_all_boosts()
        assert len(boosts) == 1
        assert boosts[0]["boost"] == 2.5
    finally:
        db.close()


def test_set_user_boost_updates(db_path):
    """set_user_boost on existing key updates boost value."""
    db = MemoryDB(db_path)
    try:
        db.set_user_boost("file", 1, "x", 1.0)
        db.set_user_boost("file", 1, "x", 3.0)
        boosts = db.get_all_boosts()
        assert boosts[0]["boost"] == 3.0
    finally:
        db.close()


def test_get_boost_returns_none_for_missing(db_path):
    """get_user_boost returns None when no boost set."""
    db = MemoryDB(db_path)
    try:
        assert db.get_user_boost("file", 999) is None
    finally:
        db.close()


def test_remove_boost_deletes(db_path):
    """remove_user_boost deletes a boost record."""
    db = MemoryDB(db_path)
    try:
        db.set_user_boost("file", 1, "x", 2.0)
        db.remove_user_boost("file", 1)
        assert db.get_user_boost("file", 1) is None
    finally:
        db.close()


def test_log_access_creates_entry(db_path):
    """log_access inserts an access log entry."""
    db = MemoryDB(db_path)
    try:
        db.log_access("file", 1, "disk_access")
        entries = db.get_access_log(days=30)
        assert len(entries) >= 1
    finally:
        db.close()


def test_get_access_log_respects_days(db_path):
    """get_access_log filters by day range."""
    db = MemoryDB(db_path)
    try:
        db.log_access("file", 1, "chat_mention")
        entries = db.get_access_log(days=0)
        assert all(e["item_type"] == "file" for e in entries)
    finally:
        db.close()


def test_upsert_scored_item(db_path):
    """upsert_scored_item inserts or updates a scored item."""
    db = MemoryDB(db_path)
    try:
        db.upsert_scored_item("file", 1, 0.85, '{"recency":0.8}')
        item = db.get_scored_item("file", 1)
        assert item is not None
        assert abs(item["score"] - 0.85) < 0.01
    finally:
        db.close()


def test_get_top_scored(db_path):
    """get_top_scored returns items sorted by score desc."""
    db = MemoryDB(db_path)
    try:
        db.upsert_scored_item("file", 1, 0.5, "{}")
        db.upsert_scored_item("file", 2, 1.0, "{}")
        db.upsert_scored_item("project", 1, 0.8, "{}")
        top = db.get_top_scored(item_type="file", limit=5)
        assert len(top) == 2
        assert top[0]["item_id"] == 2  # highest score first
    finally:
        db.close()


def test_get_scored_item_nonexistent(db_path):
    """get_scored_item returns None for missing item."""
    db = MemoryDB(db_path)
    try:
        assert db.get_scored_item("file", 999) is None
    finally:
        db.close()


def test_search_scored_by_query(db_path):
    """search_scored returns items matching a text query in signals."""
    db = MemoryDB(db_path)
    try:
        db.upsert_scored_item("project", 1, 0.9, '{"name":"hermes-agent"}')
        db.upsert_scored_item("project", 2, 0.7, '{"name":"paperclip"}')
        results = db.search_scored("hermes")
        assert len(results) >= 1
    finally:
        db.close()


def test_remove_scored_items_by_type(db_path):
    """remove_scored_items_by_type cleans up for deleted files."""
    db = MemoryDB(db_path)
    try:
        db.upsert_scored_item("file", 1, 0.5, "{}")
        db.upsert_scored_item("project", 2, 0.8, "{}")
        db.remove_scored_items_by_type("file")
        assert db.get_scored_item("file", 1) is None
        assert db.get_scored_item("project", 2) is not None
    finally:
        db.close()


def test_upsert_doc_vector(db_path):
    """upsert_doc_vector stores a sparse TF-IDF vector."""
    db = MemoryDB(db_path)
    try:
        db.upsert_doc_vector(1, '{"term1":0.5,"term2":0.3}', 2)
        vec = db.get_doc_vector(1)
        assert vec is not None
        assert '"term1"' in vec["terms"]
    finally:
        db.close()


def test_get_all_doc_vector_ids(db_path):
    """get_all_doc_vector_ids returns all indexed file ids."""
    db = MemoryDB(db_path)
    try:
        db.upsert_doc_vector(1, "{}", 0)
        db.upsert_doc_vector(2, "{}", 0)
        ids = db.get_all_doc_vector_ids()
        assert 1 in ids
        assert 2 in ids
    finally:
        db.close()