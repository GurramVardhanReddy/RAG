"""
RAG Pipeline Metrics Tracker
=============================
Comprehensive metrics tracking for Explainable RAG system.
Tracks 24 metrics across 5 categories for every question.

Categories:
    A. Pipeline Performance (Time-based)
    B. Retrieval Quality Metrics
    C. Response Quality Metrics
    D. Explainability Quality Metrics
    E. Resource & Efficiency Metrics

Author: Jaswanth Vardhan
Project: Information Retrieval Using Explainable RAG
"""

import time
import json
import psutil
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import numpy as np


# ============================================================
# A. PIPELINE PERFORMANCE (TIME-BASED) METRICS
# ============================================================

class PipelineTimer:
    """Context manager for tracking execution time of pipeline stages."""

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, stage: str):
        """Start timing a pipeline stage."""
        self._start_times[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        """Stop timing a pipeline stage and return elapsed time in seconds."""
        if stage not in self._start_times:
            raise ValueError(f"Timer for '{stage}' was never started.")
        elapsed = time.perf_counter() - self._start_times[stage]
        self.timings[stage] = elapsed
        del self._start_times[stage]
        return elapsed

    def time_stage(self, stage: str):
        """Context manager for timing a stage."""
        return _TimerContext(self, stage)

    def get_timing(self, stage: str) -> Optional[float]:
        return self.timings.get(stage)

    def get_all_timings(self) -> Dict[str, float]:
        return self.timings.copy()

    def get_end_to_end_time(self) -> float:
        """Sum of all recorded stage timings."""
        return sum(self.timings.values())

    def reset(self):
        self.timings.clear()
        self._start_times.clear()


class _TimerContext:
    """Helper context manager for PipelineTimer."""

    def __init__(self, timer: PipelineTimer, stage: str):
        self.timer = timer
        self.stage = stage

    def __enter__(self):
        self.timer.start(self.stage)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.stop(self.stage)
        return False


# ============================================================
# B. RETRIEVAL QUALITY METRICS
# ============================================================

class RetrievalMetrics:
    """
    Computes retrieval quality metrics.

    References:
        - Salemi & Zamani, "Evaluating Retrieval Quality in RAG" (SIGIR 2024)
        - Joshi et al., "Robust Multi-Model RAG Pipeline" (ICAAIC 2024)
    """

    @staticmethod
    def hit_rate(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Binary score: 1 if any relevant document is in retrieved set, else 0."""
        if not relevant_ids:
            return 0.0
        return 1.0 if any(rid in retrieved_ids for rid in relevant_ids) else 0.0

    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """MRR: 1/rank of first relevant document in retrieved list."""
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        """Recall@k: Fraction of relevant docs found in top-k retrieved."""
        if not relevant_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        found = sum(1 for rid in relevant_ids if rid in top_k)
        return found / len(relevant_ids)

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        """Precision@k: Fraction of top-k retrieved docs that are relevant."""
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        relevant_set = set(relevant_ids)
        found = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return found / len(top_k)

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        """Normalized Discounted Cumulative Gain @ k (binary relevance)."""
        relevant_set = set(relevant_ids)
        top_k = retrieved_ids[:k]

        dcg = 0.0
        for i, doc_id in enumerate(top_k):
            rel = 1.0 if doc_id in relevant_set else 0.0
            dcg += rel / np.log2(i + 2)

        ideal_rels = sorted(
            [1.0 if doc_id in relevant_set else 0.0 for doc_id in top_k],
            reverse=True
        )
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def compute_all(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> Dict[str, float]:
        """Compute all retrieval metrics at once."""
        return {
            "hit_rate": RetrievalMetrics.hit_rate(retrieved_ids, relevant_ids),
            "mrr": RetrievalMetrics.mean_reciprocal_rank(retrieved_ids, relevant_ids),
            f"recall_at_{k}": RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k),
            f"precision_at_{k}": RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k),
            f"ndcg_at_{k}": RetrievalMetrics.ndcg_at_k(retrieved_ids, relevant_ids, k),
        }


# ============================================================
# C. RESPONSE QUALITY METRICS
# ============================================================

class ResponseQualityMetrics:
    """
    Computes response quality metrics using LLM-as-a-judge approach.

    References:
        - Joshi et al., "Robust Multi-Model RAG Pipeline" (ICAAIC 2024)
    """

    @staticmethod
    def correctness_score(generated_answer: str, reference_answer: str, llm_judge_fn=None) -> float:
        """Correctness: How correct is the generated answer vs reference. Scale: 0.0 to 1.0"""
        if llm_judge_fn:
            return llm_judge_fn(
                prompt=f"""Rate the correctness of the generated answer compared to the reference answer.
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}
Rate on a scale of 0.0 to 1.0 where 1.0 is perfectly correct.
Return ONLY a decimal number."""
            )
        # Fallback: Token-level F1 score
        gen_tokens = set(generated_answer.lower().split())
        ref_tokens = set(reference_answer.lower().split())
        if not gen_tokens or not ref_tokens:
            return 0.0
        common = gen_tokens & ref_tokens
        if not common:
            return 0.0
        precision = len(common) / len(gen_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def relevancy_score(query: str, generated_answer: str, retrieved_chunks: List[str], llm_judge_fn=None) -> float:
        """Relevancy: How relevant are the retrieved chunks and answer to the query."""
        if llm_judge_fn:
            context = "\n---\n".join(retrieved_chunks[:5])
            return llm_judge_fn(
                prompt=f"""Rate how relevant the retrieved context and generated answer are to the query.
Query: {query}
Retrieved Context: {context}
Generated Answer: {generated_answer}
Rate on a scale of 0.0 to 1.0 where 1.0 is perfectly relevant.
Return ONLY a decimal number."""
            )
        query_tokens = set(query.lower().split())
        answer_tokens = set(generated_answer.lower().split())
        if not query_tokens:
            return 0.0
        return min(len(query_tokens & answer_tokens) / len(query_tokens), 1.0)

    @staticmethod
    def faithfulness_score(generated_answer: str, retrieved_chunks: List[str], llm_judge_fn=None) -> float:
        """Faithfulness: Is the answer grounded in the retrieved context?"""
        if llm_judge_fn:
            context = "\n---\n".join(retrieved_chunks[:5])
            return llm_judge_fn(
                prompt=f"""Rate the faithfulness of the generated answer.
Is every claim in the answer supported by the retrieved context?
Retrieved Context: {context}
Generated Answer: {generated_answer}
Rate on a scale of 0.0 to 1.0 where 1.0 means fully grounded in context.
Return ONLY a decimal number."""
            )
        context_text = " ".join(retrieved_chunks).lower()
        context_tokens = set(context_text.split())
        answer_tokens = generated_answer.lower().split()
        if not answer_tokens:
            return 0.0
        grounded_count = sum(1 for t in answer_tokens if t in context_tokens)
        return grounded_count / len(answer_tokens)


# ============================================================
# D. EXPLAINABILITY QUALITY METRICS
# ============================================================

class ExplainabilityMetrics:
    """
    Metrics for evaluating the quality of explanations.

    References:
        - Zhao et al., "Explainability for Large Language Models: A Survey"
        - DeYoung et al., ERASER benchmark (2020)
    """

    @staticmethod
    def citation_accuracy(
        cited_sources: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        generated_answer: str
    ) -> float:
        """Citation Accuracy: Do provenance references correctly point to source content?"""
        if not cited_sources:
            return 0.0
        correct_citations = 0
        for citation in cited_sources:
            chunk_id = citation.get("chunk_id", "")
            page_num = citation.get("page_number")
            for chunk in retrieved_chunks:
                if chunk.get("chunk_id") == chunk_id or chunk.get("page_number") == page_num:
                    cited_text = citation.get("paragraph", "").lower()
                    chunk_text = chunk.get("text", "").lower()
                    if cited_text and chunk_text and cited_text[:50] in chunk_text:
                        correct_citations += 1
                        break
                    elif not cited_text:
                        correct_citations += 1
                        break
        return correct_citations / len(cited_sources)

    @staticmethod
    def provenance_completeness(
        generated_answer: str,
        cited_sources: List[Dict[str, Any]],
        num_claims: int = None
    ) -> float:
        """Provenance Completeness: Does every claim have a source citation?"""
        if num_claims is None:
            sentences = [s.strip() for s in generated_answer.split('.') if s.strip()]
            num_claims = len(sentences)
        if num_claims == 0:
            return 1.0
        return min(len(cited_sources) / num_claims, 1.0)

    @staticmethod
    def confidence_calibration(
        confidence_scores: List[float],
        actual_correctness: List[float]
    ) -> float:
        """Expected Calibration Error (ECE) - lower is better."""
        if not confidence_scores or not actual_correctness:
            return 0.0
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(confidence_scores)
        for i in range(n_bins):
            in_bin = [
                (conf, correct)
                for conf, correct in zip(confidence_scores, actual_correctness)
                if bin_boundaries[i] <= conf < bin_boundaries[i + 1]
            ]
            if not in_bin:
                continue
            bin_size = len(in_bin)
            avg_confidence = np.mean([c for c, _ in in_bin])
            avg_accuracy = np.mean([a for _, a in in_bin])
            ece += (bin_size / total) * abs(avg_accuracy - avg_confidence)
        return ece


# ============================================================
# E. RESOURCE & EFFICIENCY METRICS
# ============================================================

class ResourceMetrics:
    """Tracks resource consumption per query."""

    def __init__(self):
        self._memory_before: Optional[float] = None

    def start_memory_tracking(self):
        """Call before processing a query."""
        self._memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    def get_memory_used(self) -> float:
        """Returns memory consumed during query processing in MB."""
        if self._memory_before is None:
            return 0.0
        current = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        return current - self._memory_before

    @staticmethod
    def count_tokens(text: str) -> int:
        """Approximate token count (~4 chars per token for English)."""
        return len(text) // 4

    @staticmethod
    def compute_context_usage(retrieved_chunks: List[str]) -> Dict[str, Any]:
        """Compute context window usage metrics."""
        combined_text = " ".join(retrieved_chunks)
        return {
            "total_tokens_in_prompt": ResourceMetrics.count_tokens(combined_text),
            "num_chunks_retrieved": len(retrieved_chunks),
            "avg_chunk_size_tokens": (
                ResourceMetrics.count_tokens(combined_text) // len(retrieved_chunks)
                if retrieved_chunks else 0
            ),
        }


# ============================================================
# MAIN METRICS RECORD
# ============================================================

@dataclass
class QueryMetricsRecord:
    """Complete metrics record for a single query."""

    # Metadata
    query_id: str = ""
    query_text: str = ""
    timestamp: str = ""

    # A. Pipeline Performance (seconds)
    document_loading_time: float = 0.0
    chunking_time: float = 0.0
    embedding_time: float = 0.0
    query_to_vector_time: float = 0.0
    similarity_search_time: float = 0.0
    generation_time: float = 0.0
    explainability_generation_time: float = 0.0
    reranking_time: float = 0.0
    end_to_end_time: float = 0.0

    # B. Retrieval Quality
    hit_rate: float = 0.0
    mrr: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    ndcg_at_k: float = 0.0

    # C. Response Quality
    correctness_score: float = 0.0
    relevancy_score: float = 0.0
    faithfulness_score: float = 0.0

    # D. Explainability Quality
    citation_accuracy: float = 0.0
    provenance_completeness: float = 0.0
    confidence_calibration_ece: float = 0.0

    # E. Resource & Efficiency
    memory_used_mb: float = 0.0
    token_count_prompt: int = 0
    chunks_retrieved: int = 0
    chunks_used_in_prompt: int = 0
    cost_per_query: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# MAIN TRACKER - TIES EVERYTHING TOGETHER
# ============================================================

class RAGMetricsTracker:
    """
    Main metrics tracker that integrates into the FastAPI RAG pipeline.

    Usage:
        tracker = RAGMetricsTracker(output_dir="./metrics_output")

        # For each query:
        record = tracker.start_query("What is the NAV of the fund?")
        # ... run pipeline stages with tracker.timer ...
        tracker.finish_query(record)

        # Export:
        tracker.export_results()
    """

    def __init__(self, output_dir: str = "./metrics_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.timer = PipelineTimer()
        self.retrieval_metrics = RetrievalMetrics()
        self.response_metrics = ResponseQualityMetrics()
        self.explainability_metrics = ExplainabilityMetrics()
        self.resource_metrics = ResourceMetrics()

        self.records: List[QueryMetricsRecord] = []

    def start_query(self, query_text: str) -> QueryMetricsRecord:
        """Initialize a new metrics record for a query."""
        self.timer.reset()
        self.resource_metrics.start_memory_tracking()
        self.timer.start("end_to_end")

        return QueryMetricsRecord(
            query_id=str(uuid.uuid4())[:8],
            query_text=query_text,
            timestamp=datetime.now().isoformat(),
        )

    def finish_query(self, record: QueryMetricsRecord) -> QueryMetricsRecord:
        """Finalize metrics for a query and store the record."""
        if "end_to_end" in self.timer._start_times:
            self.timer.stop("end_to_end")

        # Populate timing metrics
        record.document_loading_time = self.timer.get_timing("document_loading") or 0.0
        record.chunking_time = self.timer.get_timing("chunking") or 0.0
        record.embedding_time = self.timer.get_timing("embedding") or 0.0
        record.query_to_vector_time = self.timer.get_timing("query_to_vector") or 0.0
        record.similarity_search_time = self.timer.get_timing("similarity_search") or 0.0
        record.generation_time = self.timer.get_timing("generation") or 0.0
        record.explainability_generation_time = self.timer.get_timing("explainability_generation") or 0.0
        record.reranking_time = self.timer.get_timing("reranking") or 0.0
        record.end_to_end_time = self.timer.get_timing("end_to_end") or 0.0

        # Resource metrics
        record.memory_used_mb = self.resource_metrics.get_memory_used()

        self.records.append(record)
        return record

    def export_results(self, filename: str = None) -> str:
        """Export all query metrics to JSON file."""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.output_dir, filename)
        data = {
            "total_queries": len(self.records),
            "export_timestamp": datetime.now().isoformat(),
            "summary": self._compute_summary(),
            "records": [r.to_dict() for r in self.records],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def export_csv(self, filename: str = None) -> str:
        """Export all query metrics to CSV file."""
        import csv
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        if not self.records:
            return filepath
        fieldnames = list(self.records[0].to_dict().keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(record.to_dict())
        return filepath

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all queries."""
        if not self.records:
            return {}

        def _avg(field: str) -> float:
            values = [getattr(r, field) for r in self.records]
            return sum(values) / len(values) if values else 0.0

        def _min_max(field: str) -> Dict[str, float]:
            values = [getattr(r, field) for r in self.records]
            return {"min": min(values), "max": max(values), "avg": _avg(field)}

        return {
            "pipeline_performance": {
                "avg_end_to_end_time": _avg("end_to_end_time"),
                "avg_query_to_vector_time": _avg("query_to_vector_time"),
                "avg_similarity_search_time": _avg("similarity_search_time"),
                "avg_generation_time": _avg("generation_time"),
                "avg_explainability_generation_time": _avg("explainability_generation_time"),
                "end_to_end_stats": _min_max("end_to_end_time"),
            },
            "retrieval_quality": {
                "avg_hit_rate": _avg("hit_rate"),
                "avg_mrr": _avg("mrr"),
                "avg_recall_at_k": _avg("recall_at_k"),
                "avg_precision_at_k": _avg("precision_at_k"),
                "avg_ndcg_at_k": _avg("ndcg_at_k"),
            },
            "response_quality": {
                "avg_correctness": _avg("correctness_score"),
                "avg_relevancy": _avg("relevancy_score"),
                "avg_faithfulness": _avg("faithfulness_score"),
            },
            "explainability_quality": {
                "avg_citation_accuracy": _avg("citation_accuracy"),
                "avg_provenance_completeness": _avg("provenance_completeness"),
            },
            "resource_usage": {
                "avg_memory_mb": _avg("memory_used_mb"),
                "avg_token_count": _avg("token_count_prompt"),
                "avg_chunks_retrieved": _avg("chunks_retrieved"),
            },
        }

    def print_summary(self):
        """Print a formatted summary of all metrics."""
        summary = self._compute_summary()
        if not summary:
            print("No metrics recorded yet.")
            return

        print("\n" + "=" * 70)
        print("  RAG PIPELINE METRICS SUMMARY")
        print("=" * 70)
        print(f"  Total Queries Processed: {len(self.records)}")
        print("-" * 70)

        perf = summary["pipeline_performance"]
        print("\n  A. PIPELINE PERFORMANCE (Average Times)")
        print(f"    Query to Vector:        {perf['avg_query_to_vector_time']:.4f}s")
        print(f"    Similarity Search:      {perf['avg_similarity_search_time']:.4f}s")
        print(f"    Generation:             {perf['avg_generation_time']:.4f}s")
        print(f"    Explainability Gen:     {perf['avg_explainability_generation_time']:.4f}s")
        print(f"    End-to-End:             {perf['avg_end_to_end_time']:.4f}s")

        ret = summary["retrieval_quality"]
        print("\n  B. RETRIEVAL QUALITY")
        print(f"    Hit Rate:               {ret['avg_hit_rate']:.4f}")
        print(f"    MRR:                    {ret['avg_mrr']:.4f}")
        print(f"    Recall@k:               {ret['avg_recall_at_k']:.4f}")
        print(f"    Precision@k:            {ret['avg_precision_at_k']:.4f}")
        print(f"    NDCG@k:                 {ret['avg_ndcg_at_k']:.4f}")

        resp = summary["response_quality"]
        print("\n  C. RESPONSE QUALITY")
        print(f"    Correctness:            {resp['avg_correctness']:.4f}")
        print(f"    Relevancy:              {resp['avg_relevancy']:.4f}")
        print(f"    Faithfulness:           {resp['avg_faithfulness']:.4f}")

        expl = summary["explainability_quality"]
        print("\n  D. EXPLAINABILITY QUALITY")
        print(f"    Citation Accuracy:      {expl['avg_citation_accuracy']:.4f}")
        print(f"    Provenance Complete:    {expl['avg_provenance_completeness']:.4f}")

        res = summary["resource_usage"]
        print("\n  E. RESOURCE & EFFICIENCY")
        print(f"    Avg Memory Used:        {res['avg_memory_mb']:.2f} MB")
        print(f"    Avg Token Count:        {res['avg_token_count']:.0f}")
        print(f"    Avg Chunks Retrieved:   {res['avg_chunks_retrieved']:.1f}")
        print("=" * 70)
