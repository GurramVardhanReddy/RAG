"""
Retrieval Module — Domain-Aware Hybrid Retrieval
=================================================
Hybrid retrieval pipeline with domain-adaptive query embedding:

    Query → [look up domain for document_id]
          → embed query with SAME model used at index time
          → Dense + Sparse parallel search in domain collection
          → RRF Fusion
          → Cross-Encoder Rerank
          → Top-K results

Key change from v1:
    The query is embedded using the DOMAIN-SPECIFIC model that was used
    to index the document, not a single global model. This guarantees
    vector space consistency between document and query embeddings.

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
    Domain-aware hybrid retrieval pipeline: Dense + Sparse → RRF → Rerank.

    The embedding model used for query encoding is selected dynamically
    based on the domain of the document being queried, ensuring the query
    vector lives in the same vector space as the indexed document chunks.
    """

    def __init__(
        self,
        vector_store,
        embedding_model=None,           # Legacy default model (used as fallback)
        sparse_model_name: str = "prithivida/Splade_PP_en_v1",
        reranker_model_name: str = "BAAI/bge-reranker-base",
        use_hybrid: bool = True,
        use_reranker: bool = True,
        prefetch_limit: int = 20,
        document_registry: dict = None,  # Shared document registry {doc_id: {...}}
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model  # Fallback only
        self.sparse_model_name = sparse_model_name
        self.reranker_model_name = reranker_model_name
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.prefetch_limit = prefetch_limit
        self.document_registry = document_registry or {}

        self._sparse_encoder = None
        self._reranker = None

    def init_sparse_encoder(self):
        """Initialise SPLADE sparse encoder."""
        from backend.rag.retrieval.sparse_encoder import get_sparse_model
        self._sparse_encoder = get_sparse_model(self.sparse_model_name)
        logger.info(f"Sparse encoder ready: {self.sparse_model_name}")

    def init_reranker(self):
        """Initialise BGE cross-encoder reranker."""
        from sentence_transformers import CrossEncoder
        self._reranker = CrossEncoder(
            self.reranker_model_name,
            max_length=512,
            device="cpu",
        )
        logger.info(f"Cross-encoder reranker ready: {self.reranker_model_name}")

    # ----------------------------------------------------------
    # DOMAIN LOOKUP
    # ----------------------------------------------------------

    def _get_domain_info(self, document_id: str) -> tuple[str, str]:
        """
        Look up domain and embedding model for a document.

        Returns:
            (domain, embedding_model_name)
        """
        if document_id and document_id in self.document_registry:
            doc = self.document_registry[document_id]
            domain = doc.get("domain", "General")
            model = doc.get("embedding_model", "BAAI/bge-m3")
            return domain, model

        # Fallback — use General domain
        return "General", "BAAI/bge-m3"

    def _get_query_embedding_model(self, model_name: str):
        """Load the correct embedding model for query encoding."""
        from backend.rag.embedding.embedder import get_embedding_model
        return get_embedding_model(model_name)

    # ----------------------------------------------------------
    # RETRIEVAL
    # ----------------------------------------------------------

    def _encode_sparse_query(self, query: str):
        """Generate sparse query vector using SPLADE."""
        from backend.rag.retrieval.sparse_encoder import encode_sparse_query
        return encode_sparse_query(query, self.sparse_model_name)

    def _rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        """Re-score results with cross-encoder."""
        if not results:
            return results
        pairs = [(query, r["text"]) for r in results]
        ce_scores = self._reranker.predict(pairs)
        for result, ce_score in zip(results, ce_scores):
            result["rrf_score"] = result["score"]
            result["reranker_score"] = float(ce_score)
            result["score"] = float(ce_score)
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
        Run domain-aware hybrid retrieval pipeline.

        Flow:
            1. Look up domain + embedding model for document_id
            2. Embed query with domain-specific model
            3. Qdrant hybrid search in domain collection (prefetch + RRF)
            4. Cross-encoder rerank
            5. Return top-K with metadata

        Args:
            query: User's natural language question
            top_k: Number of final results
            document_id: Document filter (also determines domain)

        Returns:
            Dict with:
                - results: List of result dicts
                - metadata: Pipeline info including domain + model used
        """
        start_time = time.time()
        timings = {}

        # ── 1. Domain lookup ──────────────────────────────────────────────
        domain, model_name = self._get_domain_info(document_id)
        logger.info(f"Query routing: domain='{domain}', model='{model_name}'")

        # ── 2. Encode query with domain model ──────────────────────────────
        t0 = time.time()
        query_embedding_model = self._get_query_embedding_model(model_name)
        dense_vector = query_embedding_model.embed_query(query)
        timings["dense_encode_ms"] = (time.time() - t0) * 1000

        # ── 3. Search ──────────────────────────────────────────────────────
        retrieval_method = "dense"
        fetch_k = self.prefetch_limit if self.use_reranker else top_k

        if self.use_hybrid and self._sparse_encoder is not None:
            t0 = time.time()
            sparse_vector = self._encode_sparse_query(query)
            timings["sparse_encode_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            search_results = self.vector_store.hybrid_search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=fetch_k,
                prefetch_limit=self.prefetch_limit,
                document_id=document_id,
                domain=domain,
            )
            timings["hybrid_search_ms"] = (time.time() - t0) * 1000
            retrieval_method = "hybrid_rrf"
        else:
            t0 = time.time()
            search_results = self.vector_store.search(
                query_vector=dense_vector,
                top_k=fetch_k,
                document_id=document_id,
                domain=domain,
            )
            timings["dense_search_ms"] = (time.time() - t0) * 1000

        # ── 4. Rerank ──────────────────────────────────────────────────────
        if self.use_reranker and self._reranker is not None and search_results:
            t0 = time.time()
            search_results = self._rerank(query, search_results, top_k)
            timings["rerank_ms"] = (time.time() - t0) * 1000
            retrieval_method += "_reranked"
        else:
            search_results = search_results[:top_k]

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Retrieved {len(search_results)} docs in {elapsed_ms:.1f}ms "
            f"(method={retrieval_method}, domain={domain})"
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
                # Domain-adaptive metadata (NEW)
                "domain": domain,
                "query_embedding_model": model_name,
            },
        }
