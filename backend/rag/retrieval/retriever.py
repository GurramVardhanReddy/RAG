"""
Retrieval Module

Implements the full hybrid retrieval pipeline:

    Query → Dense + Sparse parallel search → RRF Fusion → Cross-Encoder Rerank → Top-K

Pipeline stages:
    1. ENCODE: Generate dense (BGE) + sparse (SPLADE) query vectors
    2. RETRIEVE: Qdrant hybrid_search with prefetch + RRF fusion
    3. RERANK: Cross-encoder re-scoring of top candidates
    4. RETURN: Top-K results with full metadata for explainability

The cross-encoder (BAAI/bge-reranker-base) provides a second-stage
relevance signal by jointly encoding (query, passage) pairs, which
is more accurate than bi-encoder similarity but too expensive
for first-stage retrieval.

References:
    - Wang et al., "BERT-based Dense Retrievers Require Interpolation
      with BM25 for Effective Passage Retrieval" (ICTIR 2021)
    - Barnett et al., "Seven Failure Points When Engineering a RAG" (CAIN 2024)
"""

import time
from loguru import logger
from typing import Optional


class HybridRetriever:
    """
    Full hybrid retrieval pipeline: Dense + Sparse → RRF → Rerank.

    Usage:
        retriever = HybridRetriever(
            vector_store=my_store,
            embedding_model=my_embedder,
        )
        results = retriever.retrieve("What is the refund policy?", top_k=5)
    """

    def __init__(
        self,
        vector_store,
        embedding_model,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        reranker_threshold: float = 0.1,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        use_hyde: bool = False,
        dense_prefetch: int = 40,
        sparse_prefetch: int = 10,
        openai_api_key: str = "",
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.reranker_model_name = reranker_model_name
        self.reranker_threshold = reranker_threshold
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.use_hyde = use_hyde
        self.dense_prefetch = dense_prefetch
        self.sparse_prefetch = sparse_prefetch
        self.openai_api_key = openai_api_key

        self._reranker = None

    def init_reranker(self):
        """
        Initialize the cross-encoder re-ranker.
        Called at app startup (optional — can be deferred).
        """
        from sentence_transformers import CrossEncoder

        self._reranker = CrossEncoder(
            self.reranker_model_name,
            max_length=512,
            device="cpu",
        )
        logger.info(f"Cross-encoder reranker ready: {self.reranker_model_name}")

    def _encode_sparse_query(self, query: str):
        """Generate sparse query vector using BGE-M3 lexical weights."""
        from qdrant_client.models import SparseVector
        sparse_dict = self.embedding_model.embed_query_sparse(query)
        return SparseVector(
            indices=sparse_dict["indices"],
            values=sparse_dict["values"],
        )

    def _rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Re-score results using either the cross-encoder or ColBERT.

        Cross-encoder: jointly encodes (query, passage) pairs and replaces
        the RRF score only when the CE score is not significantly worse
        (controlled by reranker_threshold). This prevents the reranker from
        actively hurting retrieval when it is uncertain.

        ColBERT: uses late token-level interaction via RAGatouille.
        """
        if not results:
            return results

        pairs = [(query, r["text"]) for r in results]
        ce_scores = self._reranker.predict(pairs)

        for result, ce_score in zip(results, ce_scores):
            result["rrf_score"] = result["score"]
            ce_score_f = float(ce_score)
            result["reranker_score"] = ce_score_f
            # Only promote the CE score if it isn't significantly below the RRF score.
            # This stops the reranker actively hurting good retrievals.
            if ce_score_f > (result["score"] - self.reranker_threshold):
                result["score"] = ce_score_f
                result["retrieval_method"] = "hybrid_rrf_reranked"
            else:
                result["retrieval_method"] = "hybrid_rrf_threshold_kept"

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: str = None,
    ) -> dict:
        """
        Run the full hybrid retrieval pipeline.

        Flow:
            1. Embed query (dense + sparse in parallel)
            2. Qdrant hybrid search (prefetch + RRF fusion)
            3. Cross-encoder rerank (if enabled)
            4. Return top-K with full metadata

        Args:
            query: User's natural language question
            top_k: Number of final results
            document_id: Optional document filter

        Returns:
            Dict with:
                - results: List of result dicts
                - metadata: Pipeline timing and method info
        """
        start_time = time.time()
        retrieval_method = "dense"
        timings = {}
        hypothesis = None

        # --- Stage 0: HyDE — generate hypothesis and embed it instead of raw query ---
        search_text = query
        if self.use_hyde and self.openai_api_key:
            from backend.rag.retrieval.hyde import generate_hypothesis
            t0 = time.time()
            hypothesis = generate_hypothesis(query, api_key=self.openai_api_key)
            search_text = hypothesis
            timings["hyde_ms"] = (time.time() - t0) * 1000
            retrieval_method = "hyde_dense"

        # --- Stage 1: Encode search text (query or hypothesis) ---
        t0 = time.time()
        dense_vector = self.embedding_model.embed_query(search_text)
        timings["dense_encode_ms"] = (time.time() - t0) * 1000

        # --- Stage 2: Search ---
        if self.use_hybrid:
            # Hybrid: Dense + Sparse → RRF
            t0 = time.time()
            sparse_vector = self._encode_sparse_query(search_text)
            timings["sparse_encode_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            search_results = self.vector_store.hybrid_search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=top_k if not self.use_reranker else self.dense_prefetch,
                dense_prefetch=self.dense_prefetch,
                sparse_prefetch=self.sparse_prefetch,
                document_id=document_id,
            )
            timings["hybrid_search_ms"] = (time.time() - t0) * 1000
            retrieval_method = "hyde_hybrid_rrf" if self.use_hyde else "hybrid_rrf"
        else:
            # Fallback: Dense-only
            t0 = time.time()
            search_results = self.vector_store.search(
                query_vector=dense_vector,
                top_k=top_k if not self.use_reranker else self.dense_prefetch,
                document_id=document_id,
            )
            timings["dense_search_ms"] = (time.time() - t0) * 1000

        # --- Stage 3: Rerank ---
        if self.use_reranker and self._reranker is not None and len(search_results) > 0:
            t0 = time.time()
            search_results = self._rerank(query, search_results, top_k)
            timings["rerank_ms"] = (time.time() - t0) * 1000
            retrieval_method += "_reranked"
        else:
            # No reranking — just truncate to top_k
            search_results = search_results[:top_k]

        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Retrieved {len(search_results)} docs in {elapsed_ms:.1f}ms "
            f"(method: {retrieval_method})"
        )

        return {
            "results": search_results,
            "metadata": {
                "retrieval_method": retrieval_method,
                "total_candidates": self.dense_prefetch + self.sparse_prefetch,
                "final_count": len(search_results),
                "processing_time_ms": round(elapsed_ms, 2),
                "timings": {k: round(v, 2) for k, v in timings.items()},
                "use_hybrid": self.use_hybrid,
                "use_reranker": self.use_reranker,
                "use_hyde": self.use_hyde,
                "hypothesis": hypothesis,
                "reranker_model": self.reranker_model_name if self.use_reranker else None,
            },
        }
