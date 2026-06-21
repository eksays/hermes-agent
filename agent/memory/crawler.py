"""File-system walker for the Hermes Memory System.

Walks configured root directories, extracts file metadata, computes
fast checksums for change detection, and incrementally syncs the
index database with the real filesystem.
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Callable, List, Optional, Set

from agent.memory import config
from agent.memory.db import MemoryDB
from agent.memory.safety import is_path_excluded
from agent.memory.project_parser import (
    detect_project, parse_dependencies, detect_git_info, find_projects,
)
from agent.memory.doc_reader import extract_text, SUPPORTED_EXTENSIONS as DOC_EXTENSIONS
from agent.memory.summarizer import extractive_summarize

logger = logging.getLogger(__name__)

# Extension -> type classification
_EXT_TYPE_MAP: dict[str, str] = {
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".tsx": "code",
    ".jsx": "code",
    ".rs": "code",
    ".go": "code",
    ".java": "code",
    ".c": "code",
    ".cpp": "code",
    ".h": "code",
    ".hpp": "code",
    ".rb": "code",
    ".php": "code",
    ".swift": "code",
    ".kt": "code",
    ".scala": "code",
    ".ex": "code",
    ".exs": "code",
    ".md": "doc",
    ".txt": "doc",
    ".rst": "doc",
    ".json": "data",
    ".yaml": "data",
    ".yml": "data",
    ".toml": "data",
    ".xml": "data",
    ".csv": "data",
    ".env": "data",
    ".html": "web",
    ".css": "web",
    ".scss": "web",
    ".less": "web",
    ".sh": "script",
    ".bat": "script",
    ".ps1": "script",
    ".lock": "other",
    ".log": "other",
}


def _fast_checksum(filepath: str) -> str:
    """SHA-256 of the first 8 KB concatenated with the file size.

    This is a fast, file-size-gated checksum suitable for change
    detection: two files with identical first-8 KB content AND the
    same total size are considered unchanged.
    """
    size = os.path.getsize(filepath)
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
            hasher.update(chunk)
    except OSError:
        pass
    hasher.update(str(size).encode())
    return hasher.hexdigest()


def _classify_extension(ext: str) -> str:
    """Map a file extension to a human-friendly type category."""
    return _EXT_TYPE_MAP.get(ext.lower(), "other")


class MemoryCrawler:
    """Walk configured root directories and sync the file index.

    Parameters
    ----------
    db : MemoryDB
        The database instance to upsert into / remove from.
    roots : list of str, optional
        One or more root directories to walk. Defaults to
        ``config.DEFAULT_ROOTS``.
    exclude_patterns : set of str, optional
        Path-segment patterns to exclude (e.g. ``{"node_modules"}``).
        Defaults to ``config.DEFAULT_EXCLUDE_PATTERNS``.
    max_file_size : int
        Maximum file size in bytes. Larger files are skipped.
    on_progress : callable, optional
        A callback ``fn(current: int, total: int, path: str)`` invoked
        after each file is processed.
    """

    def __init__(
        self,
        db: MemoryDB,
        roots: Optional[List[str]] = None,
        exclude_patterns: Optional[Set[str]] = None,
        max_file_size: int = config.MAX_FILE_SIZE_BYTES,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        self.db = db
        self.roots = roots if roots is not None else list(config.DEFAULT_ROOTS)
        self.exclude_patterns = (
            exclude_patterns
            if exclude_patterns is not None
            else config.DEFAULT_EXCLUDE_PATTERNS
        )
        self.max_file_size = max_file_size
        self.on_progress = on_progress

    # ── Public API ──────────────────────────────────────────────────────────

    def crawl(self, index_documents: bool = False) -> dict:
        """Walk all root directories and sync the database.

        Parameters
        ----------
        index_documents : bool
            When True, also extract and index text content from documents
            (currently TXT, MD, PDF, DOCX) into the documents table.

        Returns
        -------
        dict with the following counts:

        * ``files_added`` — files that were newly indexed (or re-indexed
          because their checksum changed).
        * ``files_skipped`` — files that were skipped (unchanged checksum,
          excluded path, over-large, or skipped extension).
        * ``files_removed`` — previously-indexed files that no longer exist
          on disk.
        * ``documents_indexed`` — documents whose text content was extracted
          and stored (only reported when ``index_documents=True``).
        * ``errors`` — the number of files that raised an exception during
          processing.
        """
        files_added = 0
        files_skipped = 0
        files_removed = 0
        documents_indexed = 0
        errors = 0

        seen_paths: Set[str] = set()
        walk_entries: List[str] = []

        # ── First pass: discover all walkable files ──────────────────────
        for root in self.roots:
            root_abs = os.path.abspath(root)
            if not os.path.isdir(root_abs):
                logger.warning("Crawl root does not exist: %s", root_abs)
                continue

            for dirpath, dirnames, filenames in os.walk(root_abs):
                # Prune excluded directories in-place so os.walk skips them.
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not is_path_excluded(
                        os.path.join(dirpath, d), self.exclude_patterns
                    )
                ]

                for fn in filenames:
                    fpath = os.path.join(dirpath, fn)
                    if is_path_excluded(fpath, self.exclude_patterns):
                        continue

                    _ext = os.path.splitext(fn)[1]
                    if _ext.lower() in config.SKIP_EXTENSIONS:
                        continue

                    walk_entries.append(fpath)

        total = len(walk_entries)

        # ── Second pass: process each file ───────────────────────────────
        for idx, fpath in enumerate(walk_entries):
            try:
                size = os.path.getsize(fpath)
                if size > self.max_file_size:
                    files_skipped += 1
                    seen_paths.add(fpath)
                    continue

                ext = os.path.splitext(fpath)[1].lower()
                checksum = _fast_checksum(fpath)

                # Incremental: skip when checksum already matches the DB.
                if self.db.file_exists(fpath, checksum):
                    files_skipped += 1
                    seen_paths.add(fpath)
                    continue

                filename = os.path.basename(fpath)
                modified = datetime.fromtimestamp(
                    os.path.getmtime(fpath), tz=timezone.utc
                ).isoformat()
                file_type = _classify_extension(ext)

                self.db.upsert_file(
                    path=fpath,
                    filename=filename,
                    extension=ext,
                    size_bytes=size,
                    modified_at=modified,
                    file_type=file_type,
                    checksum=checksum,
                )
                files_added += 1

                # ── Document content indexing ───────────────────────────
                if index_documents and ext in DOC_EXTENSIONS:
                    try:
                        doc_result = extract_text(fpath)
                        if doc_result is not None:
                            summary = extractive_summarize(
                                doc_result["text"], max_sentences=3
                            )
                            self.db.upsert_document(
                                path=fpath,
                                content=doc_result["text"],
                                content_hash=checksum,
                                word_count=doc_result["word_count"],
                                summary=summary,
                            )
                            documents_indexed += 1
                    except Exception:
                        logger.exception(
                            "Error indexing document content: %s", fpath
                        )
                        errors += 1

                seen_paths.add(fpath)
            except Exception:
                logger.exception("Error processing file: %s", fpath)
                errors += 1
            finally:
                if self.on_progress:
                    self.on_progress(idx + 1, total, fpath)

        # ── Stale document cleanup ──────────────────────────────────────
        if index_documents:
            for doc_path in self.db.get_all_document_paths():
                if not os.path.isfile(doc_path):
                    try:
                        self.db.remove_document(doc_path)
                    except Exception:
                        logger.exception("Error removing stale document: %s", doc_path)
                        errors += 1

        # ── Stale file cleanup ──────────────────────────────────────────
        root_abses = {os.path.abspath(r) for r in self.roots}
        for db_path in self.db.get_all_file_paths():
            if any(db_path.startswith(root_abs) for root_abs in root_abses):
                if db_path not in seen_paths:
                    try:
                        self.db.remove_file(db_path)
                        files_removed += 1
                    except Exception:
                        logger.exception(
                            "Error removing stale entry: %s", db_path
                        )
                        errors += 1

        return {
            "files_added": files_added,
            "files_skipped": files_skipped,
            "files_removed": files_removed,
            "documents_indexed": documents_indexed,
            "errors": errors,
        }

    def crawl_projects(self) -> dict:
        """Scan root directories for projects and index them.

        Returns a dict with ``projects_found``, ``projects_indexed`` counts.
        """
        projects_added = 0
        projects_found = 0

        project_roots = find_projects(self.roots, self.exclude_patterns)

        for root_path in project_roots:
            projects_found += 1
            info = detect_project(root_path)
            if info is None:
                continue

            pid = self.db.upsert_project(
                root_path=info.root_path,
                name=info.name,
                project_type=info.project_type,
                framework=info.framework,
                build_tool=info.build_tool,
            )

            # Index dependencies
            self.db.clear_project_deps(pid)
            for dep in parse_dependencies(info.project_type, root_path):
                self.db.add_project_dep(pid, dep.name, dep.version, dep.is_dev, dep.dep_type)

            # Index git info
            git_info = detect_git_info(root_path)
            if git_info is not None:
                self.db.upsert_git_repo(
                    project_id=pid,
                    git_path=git_info.git_path,
                    default_branch=git_info.default_branch,
                    remote_url=git_info.remote_url,
                    last_commit_hash=git_info.last_commit_hash,
                    last_commit_date=git_info.last_commit_date,
                    last_commit_message=git_info.last_commit_message,
                    commit_count=git_info.commit_count,
                    branch_count=git_info.branch_count,
                )

            projects_added += 1

        # Clean up stale projects (removed from disk)
        indexed_roots = set(self.db.get_all_project_paths())
        current_roots = set(project_roots)
        for stale in indexed_roots - current_roots:
            self.db.remove_project(stale)

        return {"projects_found": projects_found, "projects_indexed": projects_added}
