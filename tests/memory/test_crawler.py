"""Tests for agent.memory.crawler — file system walker with incremental scan."""

import os
import tempfile

import pytest

from agent.memory.db import MemoryDB
from agent.memory.crawler import MemoryCrawler


@pytest.fixture
def db_path() -> str:
    """Provide a temporary database path, cleaned up after the test."""
    tmp = tempfile.mktemp(suffix=".db", prefix="hermes_test_")
    yield tmp
    if os.path.exists(tmp):
        os.unlink(tmp)


def test_crawl_directory_scans_files(db_path):
    """Crawl finds Python files in a temporary directory."""
    db = MemoryDB(db_path)
    try:
        with tempfile.TemporaryDirectory(prefix="hermes_crawl_") as root:
            # Create test files
            _create_file(root, "main.py", b"print('hello')")
            _create_file(root, "utils.py", b"def util(): pass")
            _create_file(root, "README.md", b"# Docs")

            crawler = MemoryCrawler(db, roots=[root])
            result = crawler.crawl()

            assert result["files_added"] == 3
            assert result["files_skipped"] == 0
            assert result["files_removed"] == 0
            assert result["errors"] == 0

            stats = db.get_stats()
            assert stats["total_files"] == 3
    finally:
        db.close()


def test_crawl_excludes_node_modules(db_path):
    """node_modules directory content is excluded from crawl."""
    db = MemoryDB(db_path)
    try:
        with tempfile.TemporaryDirectory(prefix="hermes_crawl_") as root:
            _create_file(root, "src/main.py", b"print('hello')")

            node_modules = os.path.join(root, "node_modules", "lodash")
            os.makedirs(node_modules)
            _create_file(root, "node_modules/lodash/index.js", b"module.exports = {}")
            _create_file(root, "node_modules/lodash/LICENSE", b"MIT")

            crawler = MemoryCrawler(
                db,
                roots=[root],
                exclude_patterns={"node_modules"},
            )
            result = crawler.crawl()

            assert result["files_added"] == 1  # only src/main.py
            assert result["files_skipped"] == 0
            assert result["errors"] == 0

            stats = db.get_stats()
            assert stats["total_files"] == 1
    finally:
        db.close()


def test_incremental_scan_skips_unchanged(db_path):
    """Second crawl with no changes adds 0 files (all checksums match)."""
    db = MemoryDB(db_path)
    try:
        with tempfile.TemporaryDirectory(prefix="hermes_crawl_") as root:
            _create_file(root, "main.py", b"print('hello')")

            crawler = MemoryCrawler(db, roots=[root])

            # First crawl — should add files
            result1 = crawler.crawl()
            assert result1["files_added"] == 1

            # Second crawl — no changes, everything skipped
            result2 = crawler.crawl()
            assert result2["files_added"] == 0
            assert result2["files_skipped"] == 1  # skipped by checksum
            assert result2["files_removed"] == 0
            assert result2["errors"] == 0

            stats = db.get_stats()
            assert stats["total_files"] == 1
    finally:
        db.close()


def test_crawl_removes_stale_entries(db_path):
    """File deleted from disk is removed from DB on next crawl."""
    db = MemoryDB(db_path)
    try:
        with tempfile.TemporaryDirectory(prefix="hermes_crawl_") as root:
            _create_file(root, "main.py", b"print('hello')")
            _create_file(root, "temp.py", b"x = 1")

            crawler = MemoryCrawler(db, roots=[root])
            result1 = crawler.crawl()
            assert result1["files_added"] == 2

            # Delete temp.py from disk
            os.remove(os.path.join(root, "temp.py"))

            result2 = crawler.crawl()
            assert result2["files_added"] == 0
            assert result2["files_removed"] == 1  # temp.py cleaned up
            assert result2["errors"] == 0

            stats = db.get_stats()
            assert stats["total_files"] == 1
    finally:
        db.close()


def test_crawl_skips_binary_extensions(db_path):
    """Files with extensions in SKIP_EXTENSIONS are not indexed."""
    db = MemoryDB(db_path)
    try:
        with tempfile.TemporaryDirectory(prefix="hermes_crawl_") as root:
            _create_file(root, "script.py", b"print('ok')")
            _create_file(root, "image.png", b"fake png")
            _create_file(root, "archive.zip", b"fake zip")

            crawler = MemoryCrawler(db, roots=[root])
            result = crawler.crawl()

            assert result["files_added"] == 1  # only .py
            assert result["errors"] == 0

            stats = db.get_stats()
            assert stats["total_files"] == 1
    finally:
        db.close()


def test_crawl_classifies_file_types(db_path):
    """Files are classified into the correct type based on extension."""
    db = MemoryDB(db_path)
    try:
        with tempfile.TemporaryDirectory(prefix="hermes_crawl_") as root:
            _create_file(root, "main.py", b"code")
            _create_file(root, "readme.md", b"docs")
            _create_file(root, "data.json", b"{}")
            _create_file(root, "index.html", b"<html>")
            _create_file(root, "deploy.sh", b"#!/bin/bash")

            crawler = MemoryCrawler(db, roots=[root])
            result = crawler.crawl()
            assert result["files_added"] == 5

            for path, expected_type in [
                ("main.py", "code"),
                ("readme.md", "doc"),
                ("data.json", "data"),
                ("index.html", "web"),
                ("deploy.sh", "script"),
            ]:
                record = db.get_file_by_path(os.path.join(root, path))
                assert record is not None, f"Missing record for {path}"
                assert record["file_type"] == expected_type, (
                    f"Expected {expected_type} for {path}, got {record['file_type']}"
                )
    finally:
        db.close()


def test_crawl_skips_overlarge_files(db_path):
    """Files exceeding max_file_size are skipped."""
    db = MemoryDB(db_path)
    try:
        with tempfile.TemporaryDirectory(prefix="hermes_crawl_") as root:
            _create_file(root, "small.py", b"small")
            # Create a file larger than the limit
            large_path = os.path.join(root, "large.py")
            with open(large_path, "wb") as f:
                f.write(b"X" * 100)

            crawler = MemoryCrawler(db, roots=[root], max_file_size=50)
            result = crawler.crawl()

            assert result["files_added"] == 1  # only small.py
            assert result["errors"] == 0

            stats = db.get_stats()
            assert stats["total_files"] == 1
    finally:
        db.close()


def test_crawl_multiple_roots(db_path):
    """Crawler can walk multiple root directories."""
    db = MemoryDB(db_path)
    try:
        with (
            tempfile.TemporaryDirectory(prefix="hermes_root1_") as root1,
            tempfile.TemporaryDirectory(prefix="hermes_root2_") as root2,
        ):
            _create_file(root1, "a.py", b"a")
            _create_file(root2, "b.py", b"b")
            _create_file(root2, "c.py", b"c")

            crawler = MemoryCrawler(db, roots=[root1, root2])
            result = crawler.crawl()

            assert result["files_added"] == 3
            assert result["errors"] == 0

            stats = db.get_stats()
            assert stats["total_files"] == 3
    finally:
        db.close()


def test_crawl_on_progress_callback(db_path):
    """on_progress callback is invoked during crawl."""
    db = MemoryDB(db_path)
    calls = []

    def _progress(current: int, total: int, path: str) -> None:
        calls.append((current, total, path))

    try:
        with tempfile.TemporaryDirectory(prefix="hermes_crawl_") as root:
            _create_file(root, "a.py", b"a")
            _create_file(root, "b.py", b"b")
            _create_file(root, "c.py", b"c")

            crawler = MemoryCrawler(db, roots=[root], on_progress=_progress)
            crawler.crawl()

            assert len(calls) >= 3
            # Last call should have current == total
            last = calls[-1]
            assert last[0] == last[1]
    finally:
        db.close()


# ── Helpers ──────────────────────────────────────────────────────────────


def _create_file(root: str, rel_path: str, content: bytes) -> str:
    """Create a file at *root* / *rel_path* with *content*.

    Intermediate directories are created automatically.
    Returns the absolute path of the created file.
    """
    abspath = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(abspath), exist_ok=True)
    with open(abspath, "wb") as f:
        f.write(content)
    return abspath
