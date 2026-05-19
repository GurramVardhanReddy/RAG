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
        # Must be called once after documents are loaded
        retriever.init_sparse_encoder()

        results = retriever.retrieve("What is the refund policy?", top_k=5)
    """

    def __init__(
        self,
        vector_store,
        embedding_model,
        sparse_model_name: str = "prithivida/Splade_PP_en_v1",
        reranker_model_name: str = "BAAI/bge-reranker-base",
        use_hybrid: bool = True,
        use_reranker: bool = True,
        prefetch_limit: int = 20,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.sparse_model_name = sparse_model_name
        self.reranker_model_name = reranker_model_name
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.prefetch_limit = prefetch_limit

        # Lazy-loaded components
        self._sparse_encoder = None
        self._reranker = None

    def init_sparse_encoder(self):
        """
        Initialize the sparse encoder (SPLADE via fastembed).
        Called at app startup.
        """
        from backend.rag.retrieval.sparse_encoder import get_sparse_model

        self._sparse_encoder = get_sparse_model(self.sparse_model_name)
        logger.info(f"Sparse encoder ready: {self.sparse_model_name}")

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
        """Generate sparse query vector using SPLADE."""
        from backend.rag.retrieval.sparse_encoder import encode_sparse_query

        return encode_sparse_query(query, self.sparse_model_name)

    def _rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Re-score results using the cross-encoder.

        The cross-encoder jointly encodes (query, passage) pairs,
        providing more accurate relevance scores than bi-encoder
        cosine similarity. Results are re-sorted by cross-encoder score.

        Args:
            query: Original query text
            results: List of result dicts from hybrid search
            top_k: Number of final results to return

        Returns:
            Re-ranked and truncated results with updated scores
        """
        if not results:
            return results

        # Build (query, passage) pairs for cross-encoder
        pairs = [(query, r["text"]) for r in results]

        # Score all pairs
        ce_scores = self._reranker.predict(pairs)

        # Attach cross-encoder scores and sort descending
        for result, ce_score in zip(results, ce_scores):
            result["rrf_score"] = result["score"]  # Preserve RRF score
            result["reranker_score"] = float(ce_score)
            result["score"] = float(ce_score)  # Use CE score as primary
            result["retrieval_method"] = "hybrid_rrf_reranked"

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

        # --- Stage 1: Encode query ---
        t0 = time.time()
        dense_vector = self.embedding_model.embed_query(query)
        timings["dense_encode_ms"] = (time.time() - t0) * 1000

        # --- Stage 2: Search ---
        if self.use_hybrid and self._sparse_encoder is not None:
            # Hybrid: Dense + Sparse → RRF
            t0 = time.time()
            sparse_vector = self._encode_sparse_query(query)
            timings["sparse_encode_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            # Fetch more candidates for reranking
            fetch_k = self.prefetch_limit if self.use_reranker else top_k
            search_results = self.vector_store.hybrid_search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=fetch_k,
                prefetch_limit=self.prefetch_limit,
                document_id=document_id,
            )
            timings["hybrid_search_ms"] = (time.time() - t0) * 1000
            retrieval_method = "hybrid_rrf"
        else:
            # Fallback: Dense-only
            t0 = time.time()
            search_results = self.vector_store.search(
                query_vector=dense_vector,
                top_k=top_k if not self.use_reranker else self.prefetch_limit,
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
                "total_candidates": self.prefetch_limit,
                "final_count": len(search_results),
                "processing_time_ms": round(elapsed_ms, 2),
                "timings": {k: round(v, 2) for k, v in timings.items()},
                "use_hybrid": self.use_hybrid,
                "use_reranker": self.use_reranker,
                "sparse_model": self.sparse_model_name if self.use_hybrid else None,
                "reranker_model": self.reranker_model_name if self.use_reranker else None,
            },
        }
