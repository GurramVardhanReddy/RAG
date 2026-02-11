"""
Tests for the Explainability Engine.
"""

import pytest
from backend.rag.explainability.engine import ExplainabilityEngine


class TestExplainabilityEngine:
    """Tests for explainability features."""

    def setup_method(self):
        self.engine = ExplainabilityEngine()

    def test_confidence_high_relevance(self):
        """High cosine similarity should yield high confidence."""
        scores = [0.90, 0.88, 0.92]
        confidence = self.engine.compute_confidence_score(scores, "answer", "question")
        assert confidence >= 0.9

    def test_confidence_medium_relevance(self):
        """Medium similarity should yield medium confidence."""
        scores = [0.60, 0.55, 0.58]
        confidence = self.engine.compute_confidence_score(scores, "answer", "question")
        assert 0.4 <= confidence <= 0.8

    def test_confidence_low_relevance(self):
        """Low similarity should yield low confidence."""
        scores = [0.2, 0.15, 0.18]
        confidence = self.engine.compute_confidence_score(scores, "answer", "question")
        assert confidence <= 0.3

    def test_confidence_empty_scores(self):
        """No scores should return zero confidence."""
        confidence = self.engine.compute_confidence_score([], "answer", "question")
        assert confidence == 0.0
