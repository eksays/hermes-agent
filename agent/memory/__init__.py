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
from agent.memory.preferences import PreferenceStore
from agent.memory.scorer import Scorer
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
        self.preferences = PreferenceStore(self.db)
        self.scorer = Scorer(self.db)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._run_initial_crawl = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialize(self, blocking: bool = False) -> dict:
        """Prepare the memory index.

        By default the initial crawl is deferred to the background indexing
        thread (see ``start_background_indexing``) so that agent startup is
        never blocked by a potentially large filesystem walk. Pass
        ``blocking=True`` to run the first crawl synchronously (useful for
        tests or one-shot indexing) and receive real stats back.
        """
        logger.info("[memory] initializing at %s", self.db_path)
        if not blocking:
            # Defer the heavy walk; the background thread runs an immediate
            # first pass on start. Startup returns instantly.
            self._run_initial_crawl = True
            return {"files_added": 0, "deferred": True}
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
        # Run an immediate first pass if initialize() deferred the initial
        # crawl, so the index populates promptly without blocking startup.
        if getattr(self, "_run_initial_crawl", False):
            try:
                stats = self.crawler.crawl(index_documents=True)
                self.crawler.crawl_projects()
                logger.info("[memory] initial crawl (background): %s", stats)
            except Exception as exc:
                logger.warning("[memory] initial background crawl error: %s", exc)
            finally:
                self._run_initial_crawl = False
        while not self._stop_event.is_set():
            self._stop_event.wait(self.scan_interval_s)
            if self._stop_event.is_set():
                break
            try:
                self.crawler.crawl(index_documents=True)
                self.crawler.crawl_projects()
                # Clean up expired memory facts
                expired = self.db.cleanup_expired_facts()
                if expired:
                    logger.info("[memory] cleaned %d expired memory facts", expired)
                try:
                    score_result = self.scorer.run()
                    if score_result["scored"]:
                        logger.info("[memory] scored %d items", score_result["scored"])
                except Exception as exc:
                    logger.warning("[memory] scoring error: %s", exc)
            except Exception as exc:
                logger.warning("[memory] background index error: %s", exc)

    def shutdown(self) -> None:
        """Stop background indexing and close the database."""
        logger.info("[memory] shutting down")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.db.close()

    # ── Auto-remember ──────────────────────────────────────────────────────

    def auto_remember_from_message(self, message: str) -> int:
        """Extract and save observable facts from *message* using heuristics.

        Looks for common preference/setting patterns in user messages:

        * ``"I (like|prefer|use|want|need) ..."``
        * ``"my ... (is|are|:) ..."``
        * ``"(set|change|update|switch) ... (to|as) ..."``

        Only fires when ``auto_remember`` is enabled via config.

        Returns the number of facts saved.
        """
        if not message or not message.strip():
            return 0

        saved = 0
        lower = message.lower()

        # Pattern 1: "I like/prefer/use/want/need X"
        import re
        for match in re.finditer(
            r"\b(?:i\s+)(?:like|prefer|use|want|need|love|hate)\s+([^.!?]{3,80}?)(?=[.!?]|$)",
            lower,
        ):
            fact = match.group(1).strip()
            if len(fact) > 5:
                try:
                    self.facade.remember(
                        key=f"auto_pref_{fact[:40].replace(' ', '_')}",
                        content=f"User indicated preference: {fact}",
                        category="daily",
                        source="system",
                    )
                    saved += 1
                except Exception:
                    pass

        # Pattern 2: "my X is/are Y"
        for match in re.finditer(
            r"\bmy\s+(\w[\w\s]{1,40}?)\s+(?:is|are)\s+([^.!?]{2,60}?)(?=[.!?]|$)",
            lower,
        ):
            key = match.group(1).strip()
            val = match.group(2).strip()
            if len(key) > 1 and len(val) > 1:
                try:
                    self.facade.remember(
                        key=f"auto_{key[:40].replace(' ', '_')}",
                        content=f"User's {key}: {val}",
                        category="daily",
                        source="system",
                    )
                    saved += 1
                except Exception:
                    pass

        if saved:
            logger.info("[memory] auto-remember: saved %d facts from message", saved)
        return saved

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

    def run_scorer(self, force: bool = False) -> dict:
        """Manually trigger scoring. Returns stats."""
        return self.scorer.run()

    def tool_schemas(self) -> List[dict]:
        """Return all tool schemas for the agent registry."""
        return [
            MemoryToolsFacade.search_tool_schema(),
            MemoryToolsFacade.remember_tool_schema(),
            MemoryToolsFacade.recall_tool_schema(),
            MemoryToolsFacade.save_preference_tool_schema(),
            MemoryToolsFacade.forget_tool_schema(),
            MemoryToolsFacade.suggest_tool_schema(),
            MemoryToolsFacade.activity_tool_schema(),
            MemoryToolsFacade.stale_tool_schema(),
            MemoryToolsFacade.similar_tool_schema(),
            MemoryToolsFacade.boost_tool_schema(),
            MemoryToolsFacade.index_tool_schema(),
            MemoryToolsFacade.status_tool_schema(),
        ]
