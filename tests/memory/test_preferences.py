"""Tests for agent.memory.preferences — PreferenceStore."""

import os
import tempfile

from agent.memory.db import MemoryDB
from agent.memory.preferences import PreferenceStore


def _make_db() -> tuple[str, MemoryDB]:
    path = tempfile.mktemp(suffix=".db", prefix="hermes_test_")
    db = MemoryDB(path)
    return path, db


def _cleanup(path: str, db: MemoryDB) -> None:
    db.close()
    for f in (path, path + "-wal", path + "-shm"):
        if os.path.exists(f):
            try:
                os.unlink(f)
            except PermissionError:
                pass


def test_set_and_get():
    """set() saves and get() retrieves a preference."""
    path, db = _make_db()
    try:
        store = PreferenceStore(db)
        result = store.set("style", "concise", "behavior")
        assert result["success"] is True

        pref = store.get("style")
        assert pref is not None
        assert pref["value"] == "concise"
        assert pref["category"] == "behavior"
    finally:
        _cleanup(path, db)


def test_set_updates_existing():
    """set() with existing key updates value."""
    path, db = _make_db()
    try:
        store = PreferenceStore(db)
        store.set("k", "v1", "behavior")
        store.set("k", "v2", "style")
        pref = store.get("k")
        assert pref["value"] == "v2"
        assert pref["category"] == "style"
    finally:
        _cleanup(path, db)


def test_get_nonexistent():
    """get() on missing key returns None."""
    path, db = _make_db()
    try:
        store = PreferenceStore(db)
        assert store.get("nonexistent") is None
    finally:
        _cleanup(path, db)


def test_get_all():
    """get_all() returns all preferences."""
    path, db = _make_db()
    try:
        store = PreferenceStore(db)
        store.set("p1", "v1", "behavior")
        store.set("p2", "v2", "schedule")
        all_p = store.get_all()
        assert len(all_p) == 2
        keys = [p["key"] for p in all_p]
        assert "p1" in keys
        assert "p2" in keys
    finally:
        _cleanup(path, db)


def test_delete():
    """delete() removes a preference."""
    path, db = _make_db()
    try:
        store = PreferenceStore(db)
        store.set("del_me", "value", "behavior")
        assert store.get("del_me") is not None

        result = store.delete("del_me")
        assert result["success"] is True
        assert store.get("del_me") is None
    finally:
        _cleanup(path, db)


def test_delete_nonexistent():
    """delete() on missing key returns success=False."""
    path, db = _make_db()
    try:
        store = PreferenceStore(db)
        result = store.delete("no_such_key")
        assert result["success"] is False
    finally:
        _cleanup(path, db)


def test_get_category():
    """get_category() filters by category."""
    path, db = _make_db()
    try:
        store = PreferenceStore(db)
        store.set("s1", "v1", "behavior")
        store.set("s2", "v2", "schedule")
        store.set("s3", "v3", "behavior")

        behavior = store.get_category("behavior")
        assert len(behavior) == 2

        schedule = store.get_category("schedule")
        assert len(schedule) == 1
    finally:
        _cleanup(path, db)


def test_set_strips_pii():
    """set() strips PII from preference values."""
    path, db = _make_db()
    try:
        store = PreferenceStore(db)
        store.set("email_pref", "contact: user@example.com", "behavior")
        pref = store.get("email_pref")
        assert pref is not None
        assert "[REDACTED]" in pref["value"]
        assert "user@example.com" not in pref["value"]
    finally:
        _cleanup(path, db)
