"""Scoring pipeline for the Hermes Recommendation Engine.

Collects signals (recency, frequency, git activity, project relevance),
computes a weighted sum, and caches results in the ``scored_items`` table.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from agent.memory.db import MemoryDB
from agent.memory import config

logger = logging.getLogger(__name__)

# Helper: exponential decay score based on age in hours.
# Returns 1.0 for "now", approaches 0.0 for items older than ~30 days.
_DECAY_HALF_LIFE_H = 720  # 30 days in hours


def _decay_score(hours_old: float) -> float:
    if hours_old <= 0:
        return 1.0
    return max(0.0, 2.0 ** (-hours_old / _DECAY_HALF_LIFE_H))


class Scorer:
    """Orchestrates scoring of all indexed items.

    Parameters
    ----------
    db : MemoryDB
    weights : dict, optional
        Signal weights. Defaults to ``config.SCORING_WEIGHTS``.
    """

    def __init__(self, db: MemoryDB, weights: dict | None = None):
        self.db = db
        self.weights = weights if weights is not None else dict(config.SCORING_WEIGHTS)
        self._now = datetime.now(timezone.utc)

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Score all items and update ``scored_items``.

        Returns dict with ``scored`` (count) and ``errors`` (count).
        """
        scored = 0
        errors = 0

        for fn in (self._score_files, self._score_projects, self._score_facts):
            try:
                scored += fn()
            except Exception:
                logger.exception("Scoring error in %s", fn.__name__)
                errors += 1

        if scored:
            logger.info("[scorer] scored %d items (%d errors)", scored, errors)
        return {"scored": scored, "errors": errors}

    # ── Signal helpers ──────────────────────────────────────────────────────

    def _recency_signal(self, item_type: str, item_id: int) -> dict:
        """Exponential decay based on last modification time."""
        signals: dict = {}
        modified = None

        try:
            if item_type == "file":
                row = self.db.get_file_by_id(item_id)
                if row:
                    modified = row.get("modified_at")
            elif item_type == "project":
                row = self.db.get_project_by_id(item_id)
                if row:
                    modified = row.get("last_active")
            elif item_type == "fact":
                row = self.db.get_memory_fact_by_id(item_id)
                if row:
                    modified = row.get("updated_at")
        except Exception:
            pass

        if modified:
            try:
                dt = datetime.fromisoformat(modified)
                # Make naive datetimes offset-aware (assume UTC)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                hours = (self._now - dt).total_seconds() / 3600
                signals["recency"] = _decay_score(hours)
            except Exception:
                signals["recency"] = 0.0
        else:
            signals["recency"] = 0.0

        return signals

    def _frequency_signal(self, item_type: str, item_id: int) -> dict:
        """Score based on access frequency in last 30 days."""
        try:
            entries = self.db.get_access_log(item_type=item_type, item_id=item_id, days=30)
            count = len(entries)
            freq = 1.0 - (1.0 / (1.0 + count))
            return {"frequency": freq}
        except Exception:
            return {"frequency": 0.0}

    def _git_signal(self, project_id: int) -> dict:
        """Score based on git commit count."""
        try:
            repo = self.db.get_git_repo_by_project(project_id)
            if repo and repo.get("commit_count", 0) > 0:
                git_score = min(1.0, repo["commit_count"] / 500.0)
                return {"git_activity": git_score}
        except Exception:
            pass
        return {"git_activity": 0.0}

    def _project_signal(self, file_path: str) -> dict:
        """Score boost if file belongs to an active project."""
        try:
            for root in self.db.get_all_project_paths():
                if root and file_path.startswith(root):
                    return {"project_relevance": 0.8}
        except Exception:
            pass
        return {"project_relevance": 0.0}

    # ── Weighted sum ────────────────────────────────────────────────────────

    def _compute_score(self, signals: dict) -> float:
        score = 0.0
        for key, weight in self.weights.items():
            score += signals.get(key, 0.0) * weight
        return max(0.0, min(1.0, score))

    # ── Per-type scoring ────────────────────────────────────────────────────

    def _score_files(self) -> int:
        scored = 0
        for row in self.db.get_all_file_records():
            try:
                file_id = row["id"]
                signals: dict = {}
                signals.update(self._recency_signal("file", file_id))
                signals.update(self._frequency_signal("file", file_id))
                signals.update(self._project_signal(row.get("path", "")))
                score = self._compute_score(signals)
                boost_row = self.db.get_user_boost("file", file_id)
                boost = boost_row["boost"] if boost_row else 1.0
                self.db.upsert_scored_item(
                    "file", file_id, score * boost,
                    json.dumps(signals), boost,
                )
                scored += 1
            except Exception:
                logger.exception("Error scoring file id=%d", row.get("id"))
        return scored

    def _score_projects(self) -> int:
        scored = 0
        for row in self.db.get_all_project_records():
            try:
                pid = row["id"]
                signals: dict = {}
                signals.update(self._recency_signal("project", pid))
                signals.update(self._git_signal(pid))
                score = self._compute_score(signals)
                boost_row = self.db.get_user_boost("project", pid)
                boost = boost_row["boost"] if boost_row else 1.0
                self.db.upsert_scored_item(
                    "project", pid, score * boost,
                    json.dumps(signals), boost,
                )
                scored += 1
            except Exception:
                logger.exception("Error scoring project id=%d", row.get("id"))
        return scored

    def _score_facts(self) -> int:
        scored = 0
        for key in self.db.get_all_memory_fact_keys():
            try:
                row = self.db.get_memory_fact(key)
                if row is None:
                    continue
                signals: dict = {}
                signals.update(self._recency_signal("fact", row["id"]))
                score = self._compute_score(signals)
                boost_row = self.db.get_user_boost("fact", row["id"])
                boost = boost_row["boost"] if boost_row else 1.0
                self.db.upsert_scored_item(
                    "fact", row["id"], score * boost,
                    json.dumps(signals), boost,
                )
                scored += 1
            except Exception:
                logger.exception("Error scoring fact '%s'", key)
        return scored
