"""Preference store for the Hermes Memory System.

Provides a high-level interface for storing and retrieving user
preferences (communication style, working patterns, coding preferences,
etc.). Delegates persistence to :class:`agent.memory.db.MemoryDB`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.memory.db import MemoryDB
from agent.memory.safety import strip_pii

_MAX_VALUE_LEN = 2000


class PreferenceStore:
    """Manages user preferences persisted in SQLite.

    Preferences are read at session start and used to tailor the
    agent's behaviour, tone, and recommendations.
    """

    def __init__(self, db: MemoryDB) -> None:
        self.db = db

    def set(self, key: str, value: str, category: str = "behavior") -> Dict[str, Any]:
        """Save a preference.

        Parameters
        ----------
        key : str
            Unique preference key (e.g. ``"communication_style"``).
        value : str
            The preference value (PII is stripped automatically).
        category : str
            ``"behavior"``, ``"style"``, ``"schedule"``, or ``"stack"``.

        Returns
        -------
        dict with ``success`` and ``key``.
        """
        safe = strip_pii(value[:_MAX_VALUE_LEN])
        try:
            self.db.upsert_preference(key, safe, category)
            return {"success": True, "key": key}
        except Exception as exc:
            return {"success": False, "key": key, "error": str(exc)}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return a single preference by *key*, or ``None``."""
        return self.db.get_preference(key)

    def get_all(self) -> List[Dict[str, Any]]:
        """Return every stored preference."""
        return self.db.get_all_preferences()

    def delete(self, key: str) -> Dict[str, Any]:
        """Delete a preference.

        Returns dict with ``success`` and ``key``.
        """
        try:
            if self.db.get_preference(key) is not None:
                self.db.delete_preference(key)
                return {"success": True, "key": key}
            return {"success": False, "key": key, "error": "not found"}
        except Exception as exc:
            return {"success": False, "key": key, "error": str(exc)}

    def get_category(self, category: str) -> List[Dict[str, Any]]:
        """Return all preferences in a given category."""
        return [p for p in self.get_all() if p.get("category") == category]
