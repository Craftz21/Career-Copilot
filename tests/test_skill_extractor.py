"""Tests for skill_extractor helper functions — no DB, no embeddings."""

import pytest
from src.services.skill_extractor import _make_chunks, _alias_scan


class TestMakeChunks:
    def test_short_text_single_chunk(self):
        text = "Python FastAPI Docker"
        chunks = _make_chunks(text, chunk_size=80, overlap=20)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty(self):
        assert _make_chunks("", 80, 20) == []

    def test_long_text_produces_overlapping_chunks(self):
        words = ["word"] * 200
        text = " ".join(words)
        chunks = _make_chunks(text, chunk_size=80, overlap=20)
        # Each chunk should have at most 80 words
        for chunk in chunks:
            assert len(chunk.split()) <= 80
        # Overlap means consecutive chunks share words
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = _make_chunks(text, chunk_size=30, overlap=5)
        for chunk in chunks:
            assert len(chunk.split()) <= 30


class TestAliasScan:
    def test_finds_exact_alias(self):
        alias_map = {"python": 1, "javascript": 2, "react": 3}
        results = _alias_scan("I know python and react", alias_map)
        skill_ids = [r[0] for r in results]
        assert 1 in skill_ids  # python
        assert 3 in skill_ids  # react
        assert 2 not in skill_ids  # javascript not in text

    def test_case_insensitive_match(self):
        alias_map = {"python": 1}
        results = _alias_scan("Expert in PYTHON programming", alias_map)
        assert any(sid == 1 for sid, _ in results)

    def test_no_false_positive_short_alias(self):
        # "r" alias should not match inside "React" or "architecture"
        alias_map = {"r": 99}
        results = _alias_scan("React and architecture", alias_map)
        # Should not match (word-boundary check prevents it)
        assert 99 not in [r[0] for r in results]

    def test_confidence_is_095(self):
        alias_map = {"python": 1}
        results = _alias_scan("Python developer", alias_map)
        assert results
        assert results[0][1] == 0.95

    def test_empty_text_returns_empty(self):
        alias_map = {"python": 1}
        assert _alias_scan("", alias_map) == []
