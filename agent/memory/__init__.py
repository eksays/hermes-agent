"""Hermes Memory System — laptop-aware personal memory for the agent.

Orchestrates the memory lifecycle: init, background indexing, tool dispatch,
and shutdown. Integrates into the existing MemoryManager in agent_init.py.
"""

import os
import logging
import threading
from typing import Optional, Dict, Any, List

from agent.memory.db import MemoryDB
from agent.memory.crawler import MemoryCrawler
from agent.memory.tools import MemoryManager as MemoryToolsFacade
from agent.memory import config as mem_config

logger = logging.getLogger(__name__)


def _default_db_dir() -> str:
    hermes_home = os.environ.get("HERMES_HOME", "")
    if not hermes_home:
        hermes_home = os.path.expanduser("~/.hermes")
    return os.path.join(hermes_home, "memory")


def _default_db_path() -> str:
    return os.path.join(_default_db_dir(), "store.db")


class MemoryManager:
    """Orchestrator for the Hermes Memory System.

    Usage in agent_init.py::

        self.memory = MemoryManager()
        self.memory.start_background_indexing()
        # On each turn:
        results = self.memory.search(query)
        # On shutdown:
        self.memory.shutdown()
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        roots: Optional[List[str]] = None,
        exclude_patterns: Optional[set] = None,
        scan_interval_s: int = mem_config.DEFAULT_SCAN_INTERVAL_S,
    ):
        self.db_path = db_path or _default_db_path()
        self.roots = roots or list(mem_config.DEFAULT_ROOTS)
        self.exclude_patterns = exclude_patterns or mem_config.DEFAULT_EXCLUDE_PATTERNS
        self.scan_interval_s = scan_interval_s

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.db = MemoryDB(self.db_path)
        self.crawler = MemoryCrawler(
            self.db,
            roots=self.roots,
            exclude_patterns=self.exclude_patterns,
        )
        self.facade = MemoryToolsFacade(self.db)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialize(self) -> dict:
        """Run initial crawl + project indexing synchronously."""
        logger.info("[memory] initializing at %s", self.db_path)
        stats = self.crawler.crawl(index_documents=True)
        proj_stats = self.crawler.crawl_projects()
        stats.update(proj_stats)
        logger.info("[memory] initial crawl: %s", stats)
        return stats

    def start_background_indexing(self) -> None:
        """Start background thread for periodic re-scans."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._index_loop,
            daemon=True,
            name="memory-indexer",
        )
        self._thread.start()
        logger.info("[memory] background indexing started (interval=%ds)", self.scan_interval_s)

    def _index_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self.scan_interval_s)
            if self._stop_event.is_set():
                break
            try:
                self.crawler.crawl(index_documents=True)
                self.crawler.crawl_projects()
            except Exception as exc:
                logger.warning("[memory] background index error: %s", exc)

    def shutdown(self) -> None:
        """Stop background indexing and close the database."""
        logger.info("[memory] shutting down")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.db.close()

    # ── Tool dispatch ─────────────────────────────────────────────────────

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        return self.facade.search(query, **kwargs)

    def status(self) -> Dict[str, Any]:
        return self.facade.status()

    def force_index(self, force: bool = False) -> dict:
        """Trigger immediate re-index of files, documents and projects. Returns stats."""
        stats = self.crawler.crawl(index_documents=True)
        stats.update(self.crawler.crawl_projects())
        return stats

    def tool_schemas(self) -> List[dict]:
        """Return all tool schemas for the agent registry."""
        return [
            MemoryToolsFacade.search_tool_schema(),
            MemoryToolsFacade.index_tool_schema(),
            MemoryToolsFacade.status_tool_schema(),
        ]
