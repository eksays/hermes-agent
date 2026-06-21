"""Tests for agent.memory.scorer — scoring pipeline."""

import os
import tempfile
import json

import pytest

from agent.memory.db import MemoryDB
from agent.memory.scorer import Scorer


@pytest.fixture
def db_path():
    tmp = tempfile.mktemp(suffix=".db", prefix="hermes_test_")
    yield tmp
    if os.path.exists(tmp):
        try:
            os.unlink(tmp)
        except PermissionError:
            pass


def test_scorer_collects_recency_signal(db_path):
    """Files with recent modified_at get higher recency score."""
    db = MemoryDB(db_path)
    try:
        scorer = Scorer(db)
        signals = scorer._recency_signal("file", 0)
        assert "recency" in signals
        assert 0.0 <= signals["recency"] <= 1.0
    finally:
        db.close()


def test_scorer_weighs_and_saves(db_path):
    """run() calculates weighted scores and saves them."""
    db = MemoryDB(db_path)
    try:
        db.upsert_file("/test/old.py", "old.py", ".py", 100,
                       "2020-01-01T00:00:00", "code", "old")
        db.upsert_file("/test/new.py", "new.py", ".py", 200,
                       "2026-06-21T00:00:00", "code", "new")

        scorer = Scorer(db)
        scorer.run()

        # The newly inserted file (id=2) should have a non-zero score
        item = db.get_scored_item("file", 2)
        assert item is not None
        assert item["score"] > 0
    finally:
        db.close()


def test_scorer_respects_user_boost(db_path):
    """Boosts from user_boosts table are applied."""
    db = MemoryDB(db_path)
    try:
        db.upsert_file("/test/a.py", "a.py", ".py", 100,
                       "2026-06-21T00:00:00", "code", "a")
        db.set_user_boost("file", 1, "favorite", 3.0)

        scorer = Scorer(db)
        scorer.run()

        item = db.get_scored_item("file", 1)
        assert abs(item["boost"] - 3.0) < 0.01
    finally:
        db.close()


def test_scorer_empty_db(db_path):
    """run() on empty DB does not crash."""
    db = MemoryDB(db_path)
    try:
        scorer = Scorer(db)
        scorer.run()
    finally:
        db.close()
