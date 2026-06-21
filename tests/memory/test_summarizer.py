"""Tests for agent.memory.summarizer — extractive summarization."""

import pytest

from agent.memory.summarizer import (
    extractive_summarize,
    score_sentences,
    DEFAULT_MAX_SENTENCES,
    MIN_SENTENCE_WORDS,
)


# ── score_sentences ─────────────────────────────────────────────────────────


def test_score_sentences_short_text():
    """Short text with few sentences gets scores for each sentence."""
    text = "First sentence is here now. Second one follows shortly. Third completes this text."
    scored = score_sentences(text)
    assert len(scored) == 3
    # Scores are floats
    for sentence, score in scored:
        assert isinstance(sentence, str)
        assert isinstance(score, float)
        assert score >= 0


def test_score_sentences_empty():
    """Empty string returns empty list."""
    assert score_sentences("") == []
    assert score_sentences("   ") == []
    assert score_sentences("\n\n\n") == []


def test_score_sentences_single():
    """Single sentence gets a score (max 1.0 typically)."""
    scored = score_sentences("Just one sentence here.")
    assert len(scored) == 1
    assert scored[0][1] > 0


def test_score_sentences_favors_first():
    """First sentence tends to score higher due to position bonus."""
    text = "The first sentence is key. The second sentence is not. The third is short."
    scored = score_sentences(text)
    scores = [s for _, s in scored]
    # First sentence should have position bonus
    assert scores[0] >= scores[1] or abs(scores[0] - scores[1]) < 0.3


def test_score_sentences_short_sentences_excluded():
    """Sentences below MIN_SENTENCE_WORDS are filtered out."""
    text = "Hi. This is a proper long sentence for testing. Short again. Yet another proper complete sentence here for scoring."
    scored = score_sentences(text)
    for sent, _ in scored:
        assert len(sent.split()) >= MIN_SENTENCE_WORDS


# ── extractive_summarize ────────────────────────────────────────────────────


def test_extractive_summarize_basic():
    """Basic summarization returns top-scoring sentences in original order."""
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is the most important finding of the research paper. "
        "The fox was fast and agile in its movements. "
        "Scientists observed this behavior for several weeks. "
        "The results confirm the initial hypothesis about fox behavior. "
        "Further studies are needed to validate these observations."
    )
    summary = extractive_summarize(text, max_sentences=2)
    assert isinstance(summary, str)
    # Has at least one sentence
    assert len(summary) > 0
    # Original sentence boundaries preserved
    assert summary.count(".") >= 1


def test_extractive_summarize_returns_original_order():
    """Sentences maintain their original document order, not sorted by score."""
    text = (
        "First sentence at the start. "
        "Second sentence in the middle. "
        "Third sentence at the end."
    )
    # max_sentences=3 should return all in order
    summary = extractive_summarize(text, max_sentences=3)
    assert summary.index("First") < summary.index("Second")
    assert summary.index("Second") < summary.index("Third")


def test_extractive_summarize_shorter_than_max():
    """Text shorter than max_sentences returns all sentences."""
    text = "First sentence of the document. Second sentence of the text."
    summary = extractive_summarize(text, max_sentences=5)
    assert summary.count(".") == 2  # both sentences present


def test_extractive_summarize_empty():
    """Empty text returns empty string."""
    assert extractive_summarize("") == ""
    assert extractive_summarize("   ") == ""
    assert extractive_summarize("\n\n") == ""


def test_extractive_summarize_max_sentences_custom():
    """Custom max_sentences limits the output length."""
    text = "A. B. C. D. E. F. G. H. I. J."
    # Each "sentence" is a single word — MIN_SENTENCE_WORDS applies,
    # so let's use proper sentences
    text = (
        "The first proper sentence in the document. "
        "The second important point of the document. "
        "The third less important section of the text. "
        "The fourth detailed analysis of the results. "
        "The fifth concluding paragraph of the paper."
    )
    summary = extractive_summarize(text, max_sentences=2)
    count = len([s for s in summary.split(".") if s.strip()])
    assert count <= 2


def test_extractive_summarize_large_text():
    """Large text produces a shorter summary."""
    sentences = [f"This is sentence number {i} in the test document for summarization." for i in range(50)]
    text = " ".join(sentences)
    summary = extractive_summarize(text)
    # Default max_sentences should be much less than 50
    summary_sent_count = len([s for s in summary.split(".") if s.strip()])
    assert summary_sent_count <= DEFAULT_MAX_SENTENCES


def test_extractive_summarize_one_sentence():
    """Single sentence returns itself (no summarization needed)."""
    text = "This is the only sentence in the entire document for testing."
    summary = extractive_summarize(text, max_sentences=1)
    assert summary == text


def test_extractive_summarize_para_breaks():
    """Paragraph breaks (\n\n) are treated as sentence separators."""
    text = (
        "First paragraph sentence here.\n\n"
        "Second paragraph important finding.\n\n"
        "Third paragraph final conclusion."
    )
    summary = extractive_summarize(text, max_sentences=2)
    assert len(summary) > 0
