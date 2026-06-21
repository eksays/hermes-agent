"""TF-IDF similarity index for the Hermes Memory System.

Builds a sparse TF-IDF index over file contents and provides
cosine-similarity queries for the ``memory_similar`` tool.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter

logger = logging.getLogger(__name__)

_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "by", "with", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "when", "where", "why", "how", "if", "then", "else",
    "so", "than", "too", "very", "just", "about", "also", "into",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\'[a-z]+)?")

_FILE_TEXT_EXTS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".rst",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".css",
    ".scss", ".less", ".sh", ".bat", ".ps1", ".env", ".cfg",
    ".ini", ".conf", ".sql", ".r", ".rmd",
})

_MAX_FILE_BYTES = 10 * 1024 * 1024


def tokenize(text: str) -> list[str]:
    """Split *text* into lowercased tokens, filtering stopwords and short words."""
    if not text or not text.strip():
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def compute_tfidf(corpus: list[str]) -> list[dict[str, float]]:
    """Compute TF-IDF vectors for a list of documents.

    Returns sparse vectors where keys are terms and values are TF-IDF scores.
    """
    tokenized = [tokenize(doc) for doc in corpus]

    df: Counter = Counter()
    for doc_tokens in tokenized:
        for term in set(doc_tokens):
            df[term] += 1

    N = len(corpus)
    vectors: list[dict[str, float]] = []

    for doc_tokens in tokenized:
        if not doc_tokens:
            vectors.append({})
            continue

        tf = Counter(doc_tokens)
        max_tf = max(tf.values())

        vec: dict[str, float] = {}
        for term, count in tf.items():
            tf_norm = count / max_tf
            idf = math.log((N + 1) / (df.get(term, 0) + 1)) + 1
            vec[term] = tf_norm * idf

        vectors.append(vec)

    return vectors


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors. Returns 0.0 to 1.0."""
    if not vec_a or not vec_b:
        return 0.0

    dot = 0.0
    for term, val in vec_a.items():
        if term in vec_b:
            dot += val * vec_b[term]

    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


class TfidfIndex:
    """Sparse TF-IDF index for file-content similarity queries."""

    def __init__(self) -> None:
        self._file_vectors: list[tuple[str, dict[str, float]]] = []
        self._built = False

    def build_from_directory(self, root_path: str) -> int:
        """Scan *root_path* and build the TF-IDF index. Returns indexed file count."""
        self._file_vectors = []
        file_texts: list[str] = []
        file_paths: list[str] = []

        try:
            for dirpath, _dirnames, filenames in os.walk(root_path):
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in _FILE_TEXT_EXTS:
                        continue
                    fpath = os.path.join(dirpath, fn)
                    if os.path.getsize(fpath) > _MAX_FILE_BYTES:
                        continue
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            text = f.read()
                        file_texts.append(text)
                        file_paths.append(fpath)
                    except Exception:
                        continue
        except Exception:
            pass

        if not file_texts:
            self._built = True
            return 0

        vectors = compute_tfidf(file_texts)
        self._file_vectors = list(zip(file_paths, vectors))
        self._built = True
        return len(self._file_vectors)

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Return top-*limit* files most similar to *text*.

        Returns list of (filepath, similarity).
        """
        if not self._built or not self._file_vectors:
            return []

        query_tokens = tokenize(text)
        if not query_tokens:
            return []

        q_tf = Counter(query_tokens)
        max_q = max(q_tf.values())
        query_vec = {t: c / max_q for t, c in q_tf.items()}

        scored: list[tuple[str, float]] = []
        for fpath, doc_vec in self._file_vectors:
            sim = cosine_similarity(query_vec, doc_vec)
            if sim > 0.0:
                scored.append((fpath, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
