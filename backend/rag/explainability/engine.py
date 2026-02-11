"""
Explainability Module

The core research contribution of this project.
Provides transparency into HOW and WHY the RAG system produced its answer.

Explainability Dimensions:
1. Source Attribution  — Which chunks were used and why
2. Confidence Scoring  — How confident is the system in its answer
3. Retrieval Trace     — Full audit trail of the retrieval process
"""

from loguru import logger


class ExplainabilityEngine:
    """
    Wraps RAG pipeline outputs with explainability metadata.

    This is where your research contribution lives.
    Expand this class with the techniques you identify in your literature review.
    """

    def compute_confidence_score(
        self,
        retrieval_scores: list[float],
        answer: str,
        question: str,
    ) -> float:
        """
        Estimate confidence in the generated answer.

        Qdrant returns COSINE SIMILARITY scores (0.0 to 1.0, higher = better).
        This is the opposite of ChromaDB which returned distance.

        TODO (post literature review): Implement more sophisticated approaches:
        - LLM self-evaluation (ask the model to rate its own confidence)
        - Consistency check (generate multiple answers, measure agreement)
        - Faithfulness checking (verify answer is grounded in sources)
        - Token-level probability analysis

        Current: Heuristic based on average cosine similarity of top results.
        """
        if not retrieval_scores:
            return 0.0

        avg_score = sum(retrieval_scores) / len(retrieval_scores)

        # Cosine similarity: 1.0 = identical, 0.0 = unrelated
        if avg_score >= 0.85:
            return 0.95
        elif avg_score >= 0.70:
            return 0.80
        elif avg_score >= 0.55:
            return 0.60
        elif avg_score >= 0.40:
            return 0.40
        else:
            return 0.20
