"""
Explainability Engine — Three-Layer Framework
===============================================
The CORE RESEARCH CONTRIBUTION of this project.

Implements a novel three-layer explainability evaluation framework
specifically designed for RAG pipelines:

    Layer 1: Quantitative Explainability Metrics (Objective)
        - Source Attribution Precision & Recall
        - Provenance Completeness Score
        - Confidence Calibration Error
        - Explanation Faithfulness via Perturbation

    Layer 2: RAG-Specific Visualizations (Structural)
        - Chunk Relevance Heatmap Data
        - Provenance Trace (query → chunks → answer)
        - Confidence Distribution Data

    Layer 3: Expert Evaluation Structure (Subjective)
        - Template for domain expert ratings
        - Aggregation of expert scores

Research Gap Addressed:
    Existing RAG evaluation frameworks (RAGAS, eRAG, GaRAGe) measure
    retrieval and generation quality but NOT explanation quality.
    This module fills that gap with computable, reproducible metrics
    for evaluating how well a RAG system explains its reasoning.

References:
    - Zhang et al. (FAT* 2020) — Confidence and trust calibration
    - Rorseth et al. (RAGE) — Counterfactual provenance in RAG
    - Zhao et al. (ACM TIST 2024) — LLM explainability survey
    - Salemi & Zamani (SIGIR 2024) — eRAG retrieval evaluation

Author: Jaswanth Vardhan
Project: Information Retrieval Using Explainable RAG
"""

import re
import math
from loguru import logger
from typing import Optional
from dataclasses import dataclass, field, asdict


# ============================================================
# DATA CLASSES FOR EXPLAINABILITY OUTPUT
# ============================================================

@dataclass
class SourceAttributionMetrics:
    """Layer 1: How accurately does the system attribute answers to sources?"""
    precision: float = 0.0          # Of cited chunks, how many are relevant?
    recall: float = 0.0             # Of relevant chunks, how many were cited?
    f1_score: float = 0.0           # Harmonic mean of precision and recall
    cited_chunk_count: int = 0      # Number of chunks the system cited
    relevant_chunk_count: int = 0   # Number of chunks that are actually relevant
    correctly_cited: int = 0        # Chunks that are both cited AND relevant


@dataclass
class ProvenanceCompletenessMetrics:
    """Layer 1: How traceable is the answer back to specific sources?"""
    completeness_score: float = 0.0     # % of answer claims with source provenance
    total_claims: int = 0               # Number of distinct claims in the answer
    claims_with_provenance: int = 0     # Claims traceable to a page/paragraph
    claims_without_provenance: int = 0  # Claims that cannot be traced
    granularity_score: float = 0.0      # Avg specificity: 1.0=paragraph, 0.5=page, 0.0=none
    claim_details: list = field(default_factory=list)  # Per-claim provenance info


@dataclass
class ConfidenceCalibrationMetrics:
    """Layer 1: How well does confidence predict actual correctness?"""
    confidence_score: float = 0.0       # System's reported confidence
    calibration_bucket: str = ""        # e.g. "high (0.8-1.0)"
    confidence_tier: str = ""           # "very_high", "high", "medium", "low", "very_low"
    score_components: dict = field(default_factory=dict)  # Breakdown of how confidence was computed


@dataclass
class ExplanationFaithfulnessMetrics:
    """Layer 1: Are the cited sources actually responsible for the answer?"""
    overall_faithfulness: float = 0.0   # Avg faithfulness across cited chunks
    chunk_faithfulness: list = field(default_factory=list)  # Per-chunk faithfulness scores
    perturbation_tested: bool = False   # Whether full perturbation test was run
    answer_grounding_ratio: float = 0.0 # % of answer tokens found in cited sources


@dataclass
class ChunkRelevanceHeatmapData:
    """Layer 2: Data for rendering a chunk relevance heatmap."""
    query: str = ""
    chunks: list = field(default_factory=list)      # [{chunk_id, text_preview, page, para}]
    scores: list = field(default_factory=list)       # [float] relevance scores per chunk
    score_distribution: dict = field(default_factory=dict)  # {high: N, medium: N, low: N}
    max_score: float = 0.0
    min_score: float = 0.0
    mean_score: float = 0.0
    score_spread: float = 0.0   # max - min (higher = more discriminative retrieval)


@dataclass
class ProvenanceTraceData:
    """Layer 2: Full audit trail from query → retrieval → generation."""
    query: str = ""
    retrieval_method: str = ""
    retrieval_stages: list = field(default_factory=list)  # [{stage, description, time_ms}]
    chunk_to_answer_mapping: list = field(default_factory=list)  # [{chunk_id, answer_spans, relevance}]
    answer_with_citations: str = ""   # Answer text with inline [Source N] markers
    provenance_chain: list = field(default_factory=list)  # Full chain: query → vector → chunks → LLM → answer


@dataclass
class ExpertEvaluationTemplate:
    """Layer 3: Structure for domain expert to rate explanation quality."""
    query: str = ""
    answer: str = ""
    source_citations: list = field(default_factory=list)
    evaluation_dimensions: dict = field(default_factory=dict)
    # Dimensions: source_accuracy (1-5), confidence_reasonableness (1-5),
    #             audit_sufficiency (1-5), explanation_completeness (1-5)


@dataclass
class ThreeLayerExplainability:
    """
    Complete three-layer explainability output for a single query.
    This is the NOVEL RESEARCH CONTRIBUTION — no existing RAG framework
    provides this level of explanation evaluation.
    """
    # Layer 1: Quantitative Metrics
    source_attribution: SourceAttributionMetrics = field(default_factory=SourceAttributionMetrics)
    provenance_completeness: ProvenanceCompletenessMetrics = field(default_factory=ProvenanceCompletenessMetrics)
    confidence_calibration: ConfidenceCalibrationMetrics = field(default_factory=ConfidenceCalibrationMetrics)
    explanation_faithfulness: ExplanationFaithfulnessMetrics = field(default_factory=ExplanationFaithfulnessMetrics)

    # Layer 2: Visualization Data
    chunk_relevance_heatmap: ChunkRelevanceHeatmapData = field(default_factory=ChunkRelevanceHeatmapData)
    provenance_trace: ProvenanceTraceData = field(default_factory=ProvenanceTraceData)

    # Layer 3: Expert Evaluation
    expert_evaluation: ExpertEvaluationTemplate = field(default_factory=ExpertEvaluationTemplate)

    def to_dict(self) -> dict:
        """Serialize to dictionary for API response."""
        return asdict(self)

    def summary(self) -> dict:
        """Quick summary of key explainability scores."""
        return {
            "source_attribution_f1": round(self.source_attribution.f1_score, 4),
            "provenance_completeness": round(self.provenance_completeness.completeness_score, 4),
            "confidence_score": round(self.confidence_calibration.confidence_score, 4),
            "confidence_tier": self.confidence_calibration.confidence_tier,
            "explanation_faithfulness": round(self.explanation_faithfulness.overall_faithfulness, 4),
            "answer_grounding_ratio": round(self.explanation_faithfulness.answer_grounding_ratio, 4),
            "retrieval_discriminativeness": round(self.chunk_relevance_heatmap.score_spread, 4),
        }


# ============================================================
# EXPLAINABILITY ENGINE
# ============================================================

class ExplainabilityEngine:
    """
    Produces three-layer explainability analysis for every RAG query.

    Usage:
        engine = ExplainabilityEngine()
        result = engine.analyze(
            question="Is the Fund an exchange-traded fund?",
            answer="According to the Supplement...",
            search_results=[...],           # from retriever
            retrieval_metadata={...},       # timings, method
            total_chunks=500,
        )
        # result is a ThreeLayerExplainability dataclass
    """

    def __init__(self):
        # Relevance threshold: chunks with score above this are considered "relevant"
        self.relevance_threshold = 0.55
        # Token overlap threshold for grounding check
        self.grounding_threshold = 0.15

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================

    def analyze(
        self,
        question: str,
        answer: str,
        search_results: list[dict],
        retrieval_metadata: dict,
        total_chunks: int = 0,
    ) -> ThreeLayerExplainability:
        """
        Run full three-layer explainability analysis on a query-answer pair.

        Args:
            question: User's original question
            answer: Generated answer from GPT-4o
            search_results: List of dicts from retriever, each with:
                text, score, page_number, paragraph_number, chunk_index
            retrieval_metadata: Dict with retrieval_method, timings, etc.
            total_chunks: Total chunks in the collection

        Returns:
            ThreeLayerExplainability with all three layers populated
        """
        result = ThreeLayerExplainability()

        # --- LAYER 1: Quantitative Metrics ---
        result.source_attribution = self._compute_source_attribution(
            answer, search_results
        )
        result.provenance_completeness = self._compute_provenance_completeness(
            answer, search_results
        )
        result.confidence_calibration = self._compute_confidence_calibration(
            search_results, answer, question
        )
        result.explanation_faithfulness = self._compute_explanation_faithfulness(
            answer, search_results
        )

        # --- LAYER 2: Visualization Data ---
        result.chunk_relevance_heatmap = self._build_chunk_relevance_heatmap(
            question, search_results
        )
        result.provenance_trace = self._build_provenance_trace(
            question, answer, search_results, retrieval_metadata, total_chunks
        )

        # --- LAYER 3: Expert Evaluation Template ---
        result.expert_evaluation = self._build_expert_evaluation_template(
            question, answer, search_results
        )

        logger.info(
            f"Explainability analysis complete | "
            f"Attribution F1={result.source_attribution.f1_score:.3f} | "
            f"Provenance={result.provenance_completeness.completeness_score:.3f} | "
            f"Confidence={result.confidence_calibration.confidence_score:.3f} | "
            f"Faithfulness={result.explanation_faithfulness.overall_faithfulness:.3f}"
        )

        return result

    # ============================================================
    # LAYER 1: QUANTITATIVE EXPLAINABILITY METRICS
    # ============================================================

    def _compute_source_attribution(
        self,
        answer: str,
        search_results: list[dict],
    ) -> SourceAttributionMetrics:
        """
        Source Attribution Precision & Recall.

        Measures: Of the chunks retrieved, how many actually contributed
        to the answer? And of all chunks that could have contributed,
        how many did we retrieve?

        Method:
        - A chunk is "relevant" if its score exceeds the relevance threshold
        - A chunk is "cited" if significant token overlap exists between
          the chunk text and the generated answer
        - Precision = correctly_cited / cited_count
        - Recall = correctly_cited / relevant_count
        """
        metrics = SourceAttributionMetrics()

        if not search_results or not answer:
            return metrics

        answer_tokens = set(self._normalize_tokens(answer))
        cited_chunks = []
        relevant_chunks = []

        for i, result in enumerate(search_results):
            chunk_tokens = set(self._normalize_tokens(result.get("text", "")))
            score = result.get("score", 0.0)

            # Is this chunk relevant? (score above threshold)
            is_relevant = score >= self.relevance_threshold
            if is_relevant:
                relevant_chunks.append(i)

            # Is this chunk cited in the answer? (significant token overlap)
            if chunk_tokens and answer_tokens:
                overlap = len(answer_tokens & chunk_tokens) / len(answer_tokens)
                is_cited = overlap >= self.grounding_threshold
            else:
                is_cited = False

            if is_cited:
                cited_chunks.append(i)

        # Compute precision, recall, F1
        correctly_cited = set(cited_chunks) & set(relevant_chunks)

        metrics.cited_chunk_count = len(cited_chunks)
        metrics.relevant_chunk_count = len(relevant_chunks)
        metrics.correctly_cited = len(correctly_cited)

        if metrics.cited_chunk_count > 0:
            metrics.precision = metrics.correctly_cited / metrics.cited_chunk_count
        if metrics.relevant_chunk_count > 0:
            metrics.recall = metrics.correctly_cited / metrics.relevant_chunk_count
        if (metrics.precision + metrics.recall) > 0:
            metrics.f1_score = (
                2 * metrics.precision * metrics.recall
                / (metrics.precision + metrics.recall)
            )

        return metrics

    def _compute_provenance_completeness(
        self,
        answer: str,
        search_results: list[dict],
    ) -> ProvenanceCompletenessMetrics:
        """
        Provenance Completeness Score.

        Measures: What percentage of claims in the generated answer
        can be traced back to a specific page and paragraph?

        Method:
        - Split answer into sentences (claims)
        - For each claim, check if it overlaps with a retrieved chunk
          that has page_number and paragraph_number metadata
        - Score granularity: paragraph-level = 1.0, page-level = 0.5, none = 0.0
        """
        metrics = ProvenanceCompletenessMetrics()

        if not answer or not search_results:
            return metrics

        # Split answer into claims (sentences)
        claims = self._split_into_claims(answer)
        metrics.total_claims = len(claims)

        if not claims:
            return metrics

        granularity_scores = []

        for claim in claims:
            claim_tokens = set(self._normalize_tokens(claim))
            best_match = None
            best_overlap = 0.0

            for result in search_results:
                chunk_tokens = set(self._normalize_tokens(result.get("text", "")))
                if not chunk_tokens or not claim_tokens:
                    continue
                overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = result

            # Determine provenance level
            has_provenance = False
            granularity = 0.0

            if best_match and best_overlap >= 0.25:
                has_provenance = True
                page = best_match.get("page_number", -1)
                para = best_match.get("paragraph_number", -1)
                para_numbers = best_match.get("paragraph_numbers", [])

                if (para and para > 0) or para_numbers:
                    granularity = 1.0   # paragraph-level
                elif page is not None and page >= 0:
                    granularity = 0.5   # page-level only
                else:
                    granularity = 0.25  # chunk-level only

            if has_provenance:
                metrics.claims_with_provenance += 1
            else:
                metrics.claims_without_provenance += 1

            granularity_scores.append(granularity)

            metrics.claim_details.append({
                "claim": claim[:100],
                "has_provenance": has_provenance,
                "granularity": granularity,
                "best_overlap": round(best_overlap, 4),
                "source_page": best_match.get("page_number", -1) if best_match else -1,
                "source_para": best_match.get("paragraph_number", -1) if best_match else -1,
            })

        # Compute scores
        if metrics.total_claims > 0:
            metrics.completeness_score = metrics.claims_with_provenance / metrics.total_claims
            metrics.granularity_score = sum(granularity_scores) / len(granularity_scores)

        return metrics

    def _compute_confidence_calibration(
        self,
        search_results: list[dict],
        answer: str,
        question: str,
    ) -> ConfidenceCalibrationMetrics:
        """
        Confidence Calibration Score.

        Produces a confidence estimate and categorizes it into tiers.
        Uses a weighted combination of retrieval scores, score spread,
        answer characteristics, and coverage signals.

        The calibration error (how well confidence predicts correctness)
        is computed across multiple queries in the metrics tracker.
        """
        metrics = ConfidenceCalibrationMetrics()

        if not search_results:
            metrics.confidence_score = 0.0
            metrics.confidence_tier = "no_retrieval"
            metrics.calibration_bucket = "none"
            return metrics

        scores = [r.get("score", 0.0) for r in search_results]
        avg_score = sum(scores) / len(scores)
        top_score = max(scores)
        min_score = min(scores)
        score_spread = top_score - min_score

        # Weighted combination: 50% average, 30% top score, 20% spread
        combined = 0.5 * avg_score + 0.3 * top_score + 0.2 * (1.0 - score_spread)

        # Map to confidence tier
        if combined >= 0.82:
            confidence = 0.95
            tier = "very_high"
            bucket = "0.9-1.0"
        elif combined >= 0.68:
            confidence = 0.80
            tier = "high"
            bucket = "0.7-0.9"
        elif combined >= 0.52:
            confidence = 0.60
            tier = "medium"
            bucket = "0.5-0.7"
        elif combined >= 0.38:
            confidence = 0.40
            tier = "low"
            bucket = "0.3-0.5"
        else:
            confidence = 0.20
            tier = "very_low"
            bucket = "0.0-0.3"

        # Penalize refusal answers
        refusal_phrases = [
            "don't have enough information",
            "not found in the provided",
            "information not found",
            "cannot determine",
            "i cannot answer",
            "no relevant information",
        ]
        if any(phrase in answer.lower() for phrase in refusal_phrases):
            confidence = min(confidence, 0.15)
            tier = "refusal"
            bucket = "refusal"

        metrics.confidence_score = confidence
        metrics.confidence_tier = tier
        metrics.calibration_bucket = bucket
        metrics.score_components = {
            "avg_retrieval_score": round(avg_score, 4),
            "top_retrieval_score": round(top_score, 4),
            "min_retrieval_score": round(min_score, 4),
            "score_spread": round(score_spread, 4),
            "combined_signal": round(combined, 4),
            "num_chunks": len(scores),
        }

        return metrics

    def _compute_explanation_faithfulness(
        self,
        answer: str,
        search_results: list[dict],
    ) -> ExplanationFaithfulnessMetrics:
        """
        Explanation Faithfulness Score.

        Measures: Is the answer actually grounded in the cited sources?
        Two sub-metrics:
        1. Answer Grounding Ratio — % of answer tokens found in source chunks
        2. Per-Chunk Faithfulness — How much each chunk contributes to the answer

        Full perturbation testing (remove chunk → re-generate → check answer change)
        is available via run_perturbation_test() but disabled by default
        due to cost (requires N additional LLM calls).
        """
        metrics = ExplanationFaithfulnessMetrics()

        if not answer or not search_results:
            return metrics

        answer_tokens = self._normalize_tokens(answer)
        answer_token_set = set(answer_tokens)

        # Build combined context token set
        all_context_tokens = set()
        for r in search_results:
            all_context_tokens.update(self._normalize_tokens(r.get("text", "")))

        # Answer Grounding Ratio
        if answer_token_set:
            grounded_count = len(answer_token_set & all_context_tokens)
            metrics.answer_grounding_ratio = grounded_count / len(answer_token_set)

        # Per-chunk faithfulness: how much does each chunk contribute?
        chunk_scores = []
        for i, result in enumerate(search_results):
            chunk_tokens = set(self._normalize_tokens(result.get("text", "")))
            if not chunk_tokens or not answer_token_set:
                contribution = 0.0
            else:
                # Unique contribution: tokens in this chunk that appear in answer
                unique_to_chunk = chunk_tokens & answer_token_set
                contribution = len(unique_to_chunk) / len(answer_token_set)

            chunk_scores.append({
                "chunk_index": i,
                "page": result.get("page_number", -1),
                "para": result.get("paragraph_number", -1),
                "retrieval_score": round(result.get("score", 0.0), 4),
                "faithfulness_contribution": round(contribution, 4),
            })

        metrics.chunk_faithfulness = chunk_scores

        # Overall faithfulness is the grounding ratio (simple but effective)
        metrics.overall_faithfulness = metrics.answer_grounding_ratio
        metrics.perturbation_tested = False

        return metrics

    def run_perturbation_test(
        self,
        question: str,
        search_results: list[dict],
        original_answer: str,
        generate_fn,
    ) -> list[dict]:
        """
        OPTIONAL: Full perturbation-based faithfulness test.

        For each cited chunk, remove it from context and re-generate.
        If the answer changes significantly, that chunk was faithfully cited.

        WARNING: This makes N additional LLM calls (one per chunk).
        Only call this for evaluation runs, not production queries.

        Args:
            question: Original question
            search_results: Original retrieval results
            original_answer: The answer generated with all chunks
            generate_fn: Callable that takes (question, search_results) → answer

        Returns:
            List of dicts with per-chunk perturbation results
        """
        results = []
        original_tokens = set(self._normalize_tokens(original_answer))

        for i, chunk in enumerate(search_results):
            # Create ablated context (remove chunk i)
            ablated_results = [r for j, r in enumerate(search_results) if j != i]

            if not ablated_results:
                results.append({
                    "chunk_index": i,
                    "is_faithful": True,
                    "answer_change_ratio": 1.0,
                    "note": "Only chunk — removal produces empty context",
                })
                continue

            try:
                ablated_answer = generate_fn(question, ablated_results)
                ablated_tokens = set(self._normalize_tokens(ablated_answer))

                # Measure how much the answer changed
                if original_tokens:
                    symmetric_diff = original_tokens ^ ablated_tokens
                    change_ratio = len(symmetric_diff) / len(original_tokens | ablated_tokens)
                else:
                    change_ratio = 0.0

                # If removing this chunk changed the answer significantly,
                # it was faithfully cited (the system actually used it)
                is_faithful = change_ratio >= 0.15

                results.append({
                    "chunk_index": i,
                    "page": chunk.get("page_number", -1),
                    "para": chunk.get("paragraph_number", -1),
                    "is_faithful": is_faithful,
                    "answer_change_ratio": round(change_ratio, 4),
                })

            except Exception as e:
                logger.warning(f"Perturbation test failed for chunk {i}: {e}")
                results.append({
                    "chunk_index": i,
                    "is_faithful": None,
                    "answer_change_ratio": None,
                    "error": str(e),
                })

        return results

    # ============================================================
    # LAYER 2: RAG-SPECIFIC VISUALIZATION DATA
    # ============================================================

    def _build_chunk_relevance_heatmap(
        self,
        query: str,
        search_results: list[dict],
    ) -> ChunkRelevanceHeatmapData:
        """
        Build data for a Chunk Relevance Heatmap visualization.

        This is the RAG equivalent of an attention heatmap / saliency map.
        Shows which chunks the retriever found most relevant and their
        score distribution.
        """
        heatmap = ChunkRelevanceHeatmapData()
        heatmap.query = query

        if not search_results:
            return heatmap

        scores = []
        chunks_info = []

        for i, result in enumerate(search_results):
            score = result.get("score", 0.0)
            scores.append(score)

            chunks_info.append({
                "chunk_index": i,
                "text_preview": result.get("text", "")[:120] + "...",
                "page": result.get("page_number", -1),
                "paragraph": result.get("paragraph_number", -1),
                "score": round(score, 4),
                "relevance_band": self._score_to_band(score),
            })

        heatmap.chunks = chunks_info
        heatmap.scores = [round(s, 4) for s in scores]
        heatmap.max_score = round(max(scores), 4)
        heatmap.min_score = round(min(scores), 4)
        heatmap.mean_score = round(sum(scores) / len(scores), 4)
        heatmap.score_spread = round(max(scores) - min(scores), 4)

        # Distribution across bands
        high = sum(1 for s in scores if s >= 0.75)
        medium = sum(1 for s in scores if 0.50 <= s < 0.75)
        low = sum(1 for s in scores if s < 0.50)
        heatmap.score_distribution = {
            "high_relevance": high,
            "medium_relevance": medium,
            "low_relevance": low,
        }

        return heatmap

    def _build_provenance_trace(
        self,
        query: str,
        answer: str,
        search_results: list[dict],
        retrieval_metadata: dict,
        total_chunks: int,
    ) -> ProvenanceTraceData:
        """
        Build a complete provenance trace from query to answer.

        This is the full audit trail showing:
        Query → Encoding → Retrieval → Reranking → Generation → Answer
        with links between each retrieved chunk and the answer.
        """
        trace = ProvenanceTraceData()
        trace.query = query
        trace.retrieval_method = retrieval_metadata.get("retrieval_method", "unknown")

        # Build retrieval stage timeline
        timings = retrieval_metadata.get("timings", {})
        stages = []

        stage_descriptions = {
            "dense_encode": "Query encoded to dense vector (BGE)",
            "sparse_encode": "Query encoded to sparse vector (SPLADE)",
            "qdrant_search": "Hybrid search in Qdrant (Dense + Sparse → RRF)",
            "rerank": "Cross-encoder reranking (BGE-reranker-base)",
        }

        for stage_name, description in stage_descriptions.items():
            time_ms = timings.get(f"{stage_name}_ms", timings.get(stage_name, 0))
            if isinstance(time_ms, (int, float)):
                stages.append({
                    "stage": stage_name,
                    "description": description,
                    "time_ms": round(time_ms, 2) if time_ms else 0,
                })

        trace.retrieval_stages = stages

        # Build chunk-to-answer mapping
        answer_tokens = set(self._normalize_tokens(answer))
        mappings = []

        for i, result in enumerate(search_results):
            chunk_tokens = set(self._normalize_tokens(result.get("text", "")))
            if chunk_tokens and answer_tokens:
                overlap_tokens = chunk_tokens & answer_tokens
                overlap_ratio = len(overlap_tokens) / len(answer_tokens)
            else:
                overlap_ratio = 0.0

            mappings.append({
                "source_index": i + 1,
                "chunk_index": result.get("chunk_index", i),
                "page": result.get("page_number", -1),
                "paragraph": result.get("paragraph_number", -1),
                "retrieval_score": round(result.get("score", 0.0), 4),
                "answer_contribution": round(overlap_ratio, 4),
                "text_preview": result.get("text", "")[:150],
            })

        trace.chunk_to_answer_mapping = mappings

        # Build provenance chain summary
        trace.provenance_chain = [
            {"step": 1, "component": "User Query", "detail": query[:100]},
            {"step": 2, "component": "Hybrid Retrieval",
             "detail": f"Searched {total_chunks} chunks via {trace.retrieval_method}"},
            {"step": 3, "component": "Top-K Selection",
             "detail": f"Selected {len(search_results)} chunks (scores: {trace.chunk_to_answer_mapping[0]['retrieval_score'] if mappings else 'N/A'} to {trace.chunk_to_answer_mapping[-1]['retrieval_score'] if mappings else 'N/A'})"},
            {"step": 4, "component": "GPT-4o Generation",
             "detail": f"Generated answer from {len(search_results)} source chunks"},
            {"step": 5, "component": "Explainability Analysis",
             "detail": "Three-layer explainability metrics computed"},
        ]

        return trace

    # ============================================================
    # LAYER 3: EXPERT EVALUATION TEMPLATE
    # ============================================================

    def _build_expert_evaluation_template(
        self,
        question: str,
        answer: str,
        search_results: list[dict],
    ) -> ExpertEvaluationTemplate:
        """
        Generate a structured template for domain expert evaluation.

        The expert rates each query-answer pair on four dimensions:
        1. Source Accuracy (1-5): Are the cited sources correct?
        2. Confidence Reasonableness (1-5): Is the confidence score sensible?
        3. Audit Sufficiency (1-5): Would this satisfy a compliance audit?
        4. Explanation Completeness (1-5): Is the explanation thorough enough?

        These scores can be collected from Eoin or other domain experts
        and aggregated in the metrics tracker.
        """
        template = ExpertEvaluationTemplate()
        template.query = question
        template.answer = answer[:500]

        template.source_citations = [
            {
                "source_index": i + 1,
                "page": r.get("page_number", -1),
                "paragraph": r.get("paragraph_number", -1),
                "relevance_score": round(r.get("score", 0.0), 4),
                "text_preview": r.get("text", "")[:200],
            }
            for i, r in enumerate(search_results)
        ]

        template.evaluation_dimensions = {
            "source_accuracy": {
                "description": "Are the cited sources actually the correct sources for this answer?",
                "scale": "1 (completely wrong sources) to 5 (perfectly correct sources)",
                "score": None,  # To be filled by expert
            },
            "confidence_reasonableness": {
                "description": "Is the system's confidence score reasonable given the answer quality?",
                "scale": "1 (confidence is misleading) to 5 (confidence perfectly matches quality)",
                "score": None,
            },
            "audit_sufficiency": {
                "description": "Would this explanation (sources + confidence + provenance) satisfy a regulatory audit trail?",
                "scale": "1 (insufficient for audit) to 5 (fully audit-ready)",
                "score": None,
            },
            "explanation_completeness": {
                "description": "Does the explanation cover all aspects needed to understand the answer's derivation?",
                "scale": "1 (missing critical info) to 5 (comprehensively explained)",
                "score": None,
            },
        }

        return template

    # ============================================================
    # BACKWARD COMPATIBILITY — original methods still work
    # ============================================================

    def compute_confidence_score(
        self,
        retrieval_scores: list[float],
        answer: str,
        question: str,
    ) -> float:
        """
        Legacy method — returns just the float confidence score.
        Kept for backward compatibility with existing code.
        """
        if not retrieval_scores:
            return 0.0

        # Build minimal search_results for the new method
        fake_results = [{"score": s, "text": ""} for s in retrieval_scores]
        cal = self._compute_confidence_calibration(fake_results, answer, question)
        return cal.confidence_score

    def build_source_attribution(
        self,
        search_results: list[dict],
        retrieval_method: str = "hybrid_rrf_reranked",
    ) -> list[dict]:
        """
        Legacy method — build per-chunk source attribution.
        Kept for backward compatibility.
        """
        sources = []
        for i, result in enumerate(search_results):
            sources.append({
                "source_index": i + 1,
                "text_preview": result.get("text", "")[:300],
                "page_number": result.get("page_number", -1),
                "paragraph_number": result.get("paragraph_number", -1),
                "relevance_score": round(result.get("score", 0.0), 4),
                "retrieval_method": retrieval_method,
            })
        return sources

    # ============================================================
    # XAI TECHNIQUE: LIME ON RETRIEVAL QUERIES
    # ============================================================

    def run_lime_on_query(
        self,
        query: str,
        original_results: list[dict],
        retrieve_fn,
        top_k: int = 5,
    ) -> dict:
        """
        LIME (Local Interpretable Model-agnostic Explanations)
        applied to the retrieval pipeline.

        XAI Category: Surrogate model / feature attribution
        Reference: Ribeiro et al., KDD 2016

        For each content word in the query, removes it and re-runs
        retrieval. Measures how much the retrieved chunks change.
        Words whose removal causes the biggest change have the
        highest retrieval importance.

        This answers: "Which words in the user's question were most
        important for finding the right sources?"

        Args:
            query: Original user query
            original_results: Original retrieval results
            retrieve_fn: Callable(query, top_k) → list[dict] with 'text' and 'score'
            top_k: Number of chunks to retrieve per perturbation

        Returns:
            dict with:
              - word_importance: list of {word, index, importance, removed_query, chunk_overlap}
              - query_sensitivity: float (average importance — higher = more fragile)
              - most_important_words: top 3 words by importance
              - least_important_words: bottom 3
              - stability_score: 1 - query_sensitivity (higher = more robust retrieval)
        """
        logger.info(f"Running LIME on query: '{query}'")

        # Tokenize query into words, preserving positions
        query_words = query.split()
        if len(query_words) < 2:
            return {
                "word_importance": [],
                "query_sensitivity": 0.0,
                "most_important_words": [],
                "least_important_words": [],
                "stability_score": 1.0,
                "total_perturbations": 0,
                "note": "Query too short for LIME analysis",
            }

        # Get original chunk identifiers for comparison
        original_chunk_ids = set()
        for r in original_results:
            # Use text hash as chunk identifier
            chunk_id = hash(r.get("text", "")[:200])
            original_chunk_ids.add(chunk_id)

        original_scores = {
            hash(r.get("text", "")[:200]): r.get("score", 0.0)
            for r in original_results
        }

        # Filter to content words only (skip stopwords and short words)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'and', 'but',
            'or', 'not', 'no', 'this', 'that', 'it', 'its', 'what', 'how',
            'when', 'where', 'which', 'who', 'whom', 'if', 'then',
        }

        content_indices = []
        for i, word in enumerate(query_words):
            clean = re.sub(r'[^\w]', '', word.lower())
            if len(clean) > 2 and clean not in stopwords:
                content_indices.append(i)

        if not content_indices:
            return {
                "word_importance": [],
                "query_sensitivity": 0.0,
                "most_important_words": [],
                "least_important_words": [],
                "stability_score": 1.0,
                "total_perturbations": 0,
                "note": "No content words found in query",
            }

        # Run perturbations: remove each content word one at a time
        word_results = []

        for idx in content_indices:
            removed_word = query_words[idx]

            # Build perturbed query without this word
            perturbed_words = [w for i, w in enumerate(query_words) if i != idx]
            perturbed_query = " ".join(perturbed_words)

            if not perturbed_query.strip():
                continue

            try:
                # Re-run retrieval with perturbed query
                perturbed_results = retrieve_fn(perturbed_query, top_k)

                # Get perturbed chunk identifiers
                perturbed_chunk_ids = set()
                perturbed_scores = {}
                for r in perturbed_results:
                    chunk_id = hash(r.get("text", "")[:200])
                    perturbed_chunk_ids.add(chunk_id)
                    perturbed_scores[chunk_id] = r.get("score", 0.0)

                # Measure chunk overlap (Jaccard similarity)
                if original_chunk_ids or perturbed_chunk_ids:
                    intersection = original_chunk_ids & perturbed_chunk_ids
                    union = original_chunk_ids | perturbed_chunk_ids
                    chunk_overlap = len(intersection) / len(union) if union else 1.0
                else:
                    chunk_overlap = 1.0

                # Importance = 1 - overlap (higher = word mattered more)
                importance = 1.0 - chunk_overlap

                # Also measure average score change for overlapping chunks
                score_changes = []
                for cid in intersection:
                    orig_s = original_scores.get(cid, 0)
                    pert_s = perturbed_scores.get(cid, 0)
                    score_changes.append(abs(orig_s - pert_s))
                avg_score_change = (
                    sum(score_changes) / len(score_changes)
                    if score_changes else 0.0
                )

                word_results.append({
                    "word": removed_word,
                    "word_clean": re.sub(r'[^\w]', '', removed_word),
                    "index": idx,
                    "importance": round(importance, 4),
                    "chunk_overlap": round(chunk_overlap, 4),
                    "chunks_changed": len(union) - len(intersection),
                    "avg_score_change": round(avg_score_change, 4),
                    "removed_query": perturbed_query,
                })

            except Exception as e:
                logger.warning(f"LIME perturbation failed for word '{removed_word}': {e}")
                word_results.append({
                    "word": removed_word,
                    "word_clean": re.sub(r'[^\w]', '', removed_word),
                    "index": idx,
                    "importance": 0.0,
                    "chunk_overlap": 1.0,
                    "chunks_changed": 0,
                    "avg_score_change": 0.0,
                    "removed_query": perturbed_query,
                    "error": str(e),
                })

        # Sort by importance (highest first)
        word_results.sort(key=lambda x: x["importance"], reverse=True)

        # Compute aggregate metrics
        importances = [w["importance"] for w in word_results]
        query_sensitivity = (
            sum(importances) / len(importances)
            if importances else 0.0
        )

        most_important = [
            {"word": w["word_clean"], "importance": w["importance"]}
            for w in word_results[:3]
        ]
        least_important = [
            {"word": w["word_clean"], "importance": w["importance"]}
            for w in word_results[-3:]
        ]

        result = {
            "word_importance": word_results,
            "query_sensitivity": round(query_sensitivity, 4),
            "most_important_words": most_important,
            "least_important_words": least_important,
            "stability_score": round(1.0 - query_sensitivity, 4),
            "total_perturbations": len(word_results),
            "original_query": query,
        }

        logger.info(
            f"LIME complete | {len(word_results)} perturbations | "
            f"sensitivity={query_sensitivity:.3f} | "
            f"top word='{most_important[0]['word'] if most_important else 'N/A'}'"
        )

        return result

    # ============================================================
    # HELPER METHODS
    # ============================================================

    @staticmethod
    def _normalize_tokens(text: str) -> list[str]:
        """Tokenize and normalize text for comparison."""
        if not text:
            return []
        # Lowercase, remove punctuation, split on whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        # Remove very short tokens and common stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'and',
            'but', 'or', 'not', 'no', 'this', 'that', 'these', 'those',
            'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you',
        }
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    @staticmethod
    def _split_into_claims(text: str) -> list[str]:
        """Split answer text into individual claims (sentences)."""
        if not text:
            return []
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out very short fragments
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    @staticmethod
    def _score_to_band(score: float) -> str:
        """Categorize a retrieval score into a relevance band."""
        if score >= 0.75:
            return "high"
        elif score >= 0.50:
            return "medium"
        else:
            return "low"