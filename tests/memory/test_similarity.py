"""Tests for agent.memory.similarity — TF-IDF index and cosine similarity."""

import os
import tempfile

from agent.memory.similarity import TfidfIndex, tokenize, compute_tfidf, cosine_similarity


def test_tokenize_basic():
    """tokenize splits and lowercases words, removes stopwords."""
    tokens = tokenize("Hello World! This is a TEST.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens
    assert "a" not in tokens
    assert "is" not in tokens


def test_tokenize_empty():
    """tokenize returns empty list for empty input."""
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_tokenize_removes_stopwords():
    """Common English stopwords are filtered out."""
    tokens = tokenize("the and or for in a is it")
    assert all(t not in tokens for t in ["the", "and", "or", "for", "in", "a", "is", "it"])


def test_tokenize_short_words_removed():
    """Single character words are filtered out (length filter > 1)."""
    tokens = tokenize("a b c hello world")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "hello" in tokens


def test_compute_tfidf():
    """TF-IDF vector has expected shape and terms."""
    corpus = [
        "the quick brown fox jumps",
        "the lazy dog sleeps",
    ]
    vectors = compute_tfidf(corpus)
    assert len(vectors) == 2
    assert any("quick" in v for v in vectors)
    assert any("lazy" in v for v in vectors)


def test_cosine_similarity_identical():
    """Identical vectors have cosine = 1.0."""
    v1 = {"a": 0.5, "b": 0.5}
    v2 = {"a": 0.5, "b": 0.5}
    assert abs(cosine_similarity(v1, v2) - 1.0) < 0.001


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors have cosine = 0.0."""
    v1 = {"a": 0.5}
    v2 = {"b": 0.5}
    assert abs(cosine_similarity(v1, v2) - 0.0) < 0.001


def test_cosine_similarity_empty():
    """Empty vectors return 0.0."""
    assert cosine_similarity({}, {}) == 0.0
    assert cosine_similarity({"a": 0.5}, {}) == 0.0
    assert cosine_similarity({}, {"a": 0.5}) == 0.0


def test_tfidf_index_build_and_query(tmp_path):
    """TfidfIndex builds and finds similar items."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "readme.md").write_text("This project is a web application using Python Flask")
    (doc_dir / "setup.py").write_text("from setuptools import setup")

    index = TfidfIndex()
    index.build_from_directory(str(doc_dir))

    results = index.query("Python web application")
    assert len(results) > 0
    assert any("readme" in r[0] for r in results)


def test_tfidf_index_empty():
    """Index on empty directory does not crash."""
    index = TfidfIndex()
    index.build_from_directory("/nonexistent")
    assert index.query("test") == []
