"""Extractive text summarization for the Hermes Memory System.

Deterministic, zero-API summarization using sentence-scoring heuristics:
term frequency, position bias, and length filtering. Suitable for
producing lightweight document summaries during indexing.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

# Minimum number of words a sentence must have to be considered.
MIN_SENTENCE_WORDS = 4

# Default number of sentences to keep in the summary.
DEFAULT_MAX_SENTENCES = 5

# Regex to split text into sentences.
_SENTENCE_SPLIT = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\n)\s+")


def score_sentences(text: str) -> list[tuple[str, float]]:
    """Score each sentence in *text* by estimated importance.

    Heuristics
    ----------
    * **Term frequency** — sentences containing rarer words score higher.
    * **Position bonus** — the first sentence receives a boost.
    * **Length filter** — very short sentences (fewer than
      :const:`MIN_SENTENCE_WORDS` tokens) are excluded.

    Parameters
    ----------
    text : str
        The full document text.

    Returns
    -------
    list of (sentence, score)
        Sentences in document order with their importance score.
    """
    if not text or not text.strip():
        return []

    raw_sentences = _split_sentences(text)
    if not raw_sentences:
        return []

    # Frequency table (lower-cased words, no punctuation).
    words = re.findall(r"\w+", text.lower())
    if not words:
        return []
    freq = Counter(words)
    max_freq = max(freq.values()) if freq else 1

    scored: list[tuple[str, float]] = []
    for idx, sentence in enumerate(raw_sentences):
        sent_words = re.findall(r"\w+", sentence.lower())
        if len(sent_words) < MIN_SENTENCE_WORDS:
            continue

        # Term-frequency score (normalised).
        tf_score = sum(freq.get(w, 0) / max_freq for w in set(sent_words))
        tf_score /= max(len(set(sent_words)), 1)

        # Position bonus: first sentence gets +0.3, linearly decays.
        pos_bonus = max(0.3 - (idx * 0.02), 0.0)

        # Length penalty: very long sentences get a small penalty.
        length_penalty = min(len(sent_words) / 100.0, 0.5)

        score = tf_score + pos_bonus - length_penalty
        scored.append((sentence, max(score, 0.0)))

    return scored


def extractive_summarize(
    text: str,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
) -> str:
    """Return an extractive summary of *text*.

    Selects the top-*max_sentences* scoring sentences (by
    :func:`score_sentences`) while preserving the original document order.

    Parameters
    ----------
    text : str
        The document text to summarise.
    max_sentences : int
        Maximum number of sentences to include in the summary.

    Returns
    -------
    str
        The summary, or an empty string if *text* is empty or contains
        no sentence-long enough.
    """
    if not text or not text.strip():
        return ""

    scored = score_sentences(text)
    if not scored:
        return ""

    if len(scored) <= max_sentences:
        return " ".join(s for s, _ in scored)

    # Select top-N by score, then re-sort by original position.
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    selected = set(id(s) for s, _ in scored_sorted[:max_sentences])

    ordered = [s for s, _ in scored if id(s) in selected]
    return " ".join(ordered)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences while preserving whitespace boundaries."""
    raw = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in raw if s.strip()]
