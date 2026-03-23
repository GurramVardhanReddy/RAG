"""
Explainability Module

The core research contribution of this project.
Provides transparency into HOW and WHY the RAG system produced its answer.

Explainability Dimensions:
1. Source Attribution  — Which chunks were used and why
2. Confidence Scoring  — How confident is the system in its answer
3. Retrieval Trace     — Full audit trail of the retrieval process
4. Chunk Relevance     — Per-chunk relevance breakdown
"""

from loguru import logger
from typing import Optional


class ExplainabilityEngine:
    """
    Wraps RAG pipeline outputs with explainability metadata.
    """

    def __init__(self):
        pass

    def compute_confidence_score(
        self,
        retrieval_scores: list[float],
        answer: str,
        question: str,
    ) -> float:
        """
        Estimate confidence in the generated answer.

        Uses cosine similarity scores from Qdrant (higher = better, 0 to 1).

        Heuristic:
        - avg >= 0.85  → 0.95 confidence (very similar chunks found)
        - avg >= 0.70  → 0.80 confidence
        - avg >= 0.55  → 0.60 confidence
        - avg >= 0.40  → 0.40 confidence
        - avg <  0.40  → 0.20 confidence (poor retrieval)
        """
        if not retrieval_scores:
            return 0.0

        avg_score = sum(retrieval_scores) / len(retrieval_scores)

        # Also factor in the top score (best match matters a lot)
        top_score = max(retrieval_scores)

        # Weighted: 60% average, 40% top score
        combined = 0.6 * avg_score + 0.4 * top_score

        if combined >= 0.85:
            confidence = 0.95
        elif combined >= 0.70:
            confidence = 0.80
        elif combined >= 0.55:
            confidence = 0.60
        elif combined >= 0.40:
            confidence = 0.40
        else:
            confidence = 0.20

        # Penalize if answer is a refusal
        refusal_phrases = [
            "don't have enough information",
            "not found in the provided",
            "information not found",
            "cannot determine",
        ]
        if any(phrase in answer.lower() for phrase in refusal_phrases):
            confidence = min(confidence, 0.15)

        logger.debug(
            f"Confidence: {confidence:.2f} "
            f"(avg_score={avg_score:.4f}, top_score={top_score:.4f}, combined={combined:.4f})"
        )

        return confidence

    def build_source_attribution(
        self,
        search_results: list[dict],
        retrieval_method: str = "vector_search_cosine",
    ) -> list[dict]:
        """
        Build per-chunk source attribution data with page + paragraph numbers.
        """
        sources = []
        for i, result in enumerate(search_results):
            sources.append({
                "content": result.get("text", ""),
                "page_number": result.get("page_number", -1),
                "paragraph_number": result.get("paragraph_number", -1),
                "paragraph_numbers": result.get("paragraph_numbers", []),
                "chunk_index": result.get("chunk_index", i),
                "relevance_score": round(float(result.get("score", 0.0)), 4),
                "retrieval_method": retrieval_method,
            })
        return sources

    def build_retrieval_trace(
        self,
        query: str,
        search_results: list[dict],
        retrieval_method: str = "vector_search_cosine",
        processing_time_ms: float = 0.0,
    ) -> dict:
        """
        Build a full audit trail of the retrieval process.
        """
        scores = [r.get("score", 0.0) for r in search_results]
        return {
            "query": query,
            "retrieval_method": retrieval_method,
            "total_results": len(search_results),
            "score_range": {
                "min": round(min(scores), 4) if scores else 0.0,
                "max": round(max(scores), 4) if scores else 0.0,
                "avg": round(sum(scores) / len(scores), 4) if scores else 0.0,
            },
            "processing_time_ms": round(processing_time_ms, 2),
            "pages_covered": sorted(set(
                r.get("page_number", -1)
                for r in search_results
                if r.get("page_number", -1) >= 0
            )),
            "paragraphs_covered": sorted(set(
                r.get("paragraph_number", -1)
                for r in search_results
                if r.get("paragraph_number", -1) > 0
            )),
        }
