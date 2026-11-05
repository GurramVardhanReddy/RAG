"""
Retrieval Module — Domain-Aware Hybrid Retrieval
=================================================
Hybrid retrieval pipeline with domain-adaptive query embedding:

    Query → [look up domain for document_id]
          → embed query with SAME model used at index time
          → Dense + Sparse search in domain collection
          → RRF Fusion
          → Cross-Encoder Rerank (with threshold)
          → Top-K results

Key design:
    Query is embedded using the DOMAIN-SPECIFIC model that indexed the
    document, guaranteeing vector space consistency.
    The sparse arm uses learned weights when the model has them (BGE-M3)
    and BM25 otherwise, so every domain gets hybrid retrieval. Before the
    BM25 arm, dense-only domain models silently skipped the sparse arm.
    HyDE generates a hypothetical passage before embedding when enabled.

References:
    - Wang et al., "BERT-based Dense Retrievers Require Interpolation
      with BM25 for Effective Passage Retrieval" (ICTIR 2021)
    - Barnett et al., "Seven Failure Points When Engineering a RAG" (CAIN 2024)
"""

import time
from loguru import logger
from typing import Optional

from backend.rag.retrieval.bm25_sparse import BM25SparseEncoder

# Must match DOMAIN_MODEL_MAP["General"] in domain_detector.py
GENERAL_MODEL = "BAAI/bge-base-en-v1.5"


class HybridRetriever:
    """
    Domain-aware hybrid retrieval pipeline: Dense (+ Sparse) → RRF → Rerank.

    The embedding model for query encoding is selected dynamically based on
    the domain of the document being queried, ensuring the query vector lives
    in the same vector space as the indexed document chunks.
    """

    def __init__(
        self,
        vector_store,
        embedding_model=None,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        reranker_threshold: float = 0.1,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        use_hyde: bool = False,
        dense_prefetch: int = 40,
        sparse_prefetch: int = 10,
        openai_api_key: str = "",
        document_registry: dict = None,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model      # Fallback when no domain info
        self.reranker_model_name = reranker_model_name
        self.reranker_threshold = reranker_threshold
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.use_hyde = use_hyde
        self.dense_prefetch = dense_prefetch
        self.sparse_prefetch = sparse_prefetch
        self.openai_api_key = openai_api_key
        self.document_registry = document_registry or {}

        self._reranker = None
        self._bm25 = BM25SparseEncoder()

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
        Returns (domain, embedding_model_name).
        """
        if document_id and document_id in self.document_registry:
            doc = self.document_registry[document_id]
            domain = doc.get("domain", "General")
            model  = doc.get("embedding_model", GENERAL_MODEL)
            return domain, model
        return "General", GENERAL_MODEL

    def _get_query_embedding_model(self, model_name: str):
        """Load the correct embedding model for query encoding."""
        from backend.rag.embedding.embedder import get_embedding_model
        return get_embedding_model(model_name)

    # ----------------------------------------------------------
    # SPARSE ENCODING (learned for BGE-M3, BM25 for every other model)
    # ----------------------------------------------------------

    def _encode_sparse_query(self, query: str, embedding_model, domain: str = "General"):
        """
        Generate the sparse query vector for the domain collection.
        Returns (sparse_dict, vector_name), or (None, None) when no sparse arm exists.
        """
        if hasattr(embedding_model, "embed_query_sparse"):
            return embedding_model.embed_query_sparse(query), "sparse"
        if self.vector_store.supports_bm25(domain):
            return self._bm25.embed_query(query), "bm25"
        return None, None

    # ----------------------------------------------------------
    # RERANKING
    # ----------------------------------------------------------

    def _rerank(
        self, query: str, results: list[dict], top_k: int, method: str = "hybrid_rrf"
    ) -> list[dict]:
        """
        Re-score results with cross-encoder.
        Only replaces the RRF score if CE score is not significantly worse
        (threshold guard prevents the reranker from actively hurting results).
        """
        if not results:
            return results

        pairs = [(query, r["text"]) for r in results]
        ce_scores = self._reranker.predict(pairs)

        for result, ce_score in zip(results, ce_scores):
            result["rrf_score"] = result["score"]
            ce_score_f = float(ce_score)
            result["reranker_score"] = ce_score_f
            if ce_score_f > (result["score"] - self.reranker_threshold):
                result["score"] = ce_score_f
                result["retrieval_method"] = f"{method}_reranked"
            else:
                result["retrieval_method"] = f"{method}_threshold_kept"

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    # ----------------------------------------------------------
    # MAIN RETRIEVE
    # ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: str = None,
    ) -> dict:
        """
        Run domain-aware hybrid retrieval pipeline.

        Flow:
            1. HyDE: generate hypothesis (optional)
            2. Domain lookup → select correct embedding model
            3. Embed query with domain model
            4. Hybrid search (sparse only if model supports it)
            5. Rerank with threshold guard
            6. Return top-K with full metadata
        """
        start_time = time.time()
        timings = {}
        hypothesis = None

        # ── 0. HyDE ───────────────────────────────────────────────────────────
        search_text = query
        if self.use_hyde and self.openai_api_key:
            from backend.rag.retrieval.hyde import generate_hypothesis
            t0 = time.time()
            hypothesis = generate_hypothesis(query, api_key=self.openai_api_key)
            search_text = hypothesis
            timings["hyde_ms"] = (time.time() - t0) * 1000

        # ── 1. Domain lookup ──────────────────────────────────────────────────
        domain, model_name = self._get_domain_info(document_id)
        logger.info(f"Query routing: domain='{domain}', model='{model_name}'")

        # ── 2. Encode query ───────────────────────────────────────────────────
        t0 = time.time()
        query_model = self._get_query_embedding_model(model_name)
        dense_vector = query_model.embed_query(search_text)
        timings["dense_encode_ms"] = (time.time() - t0) * 1000

        # ── 3. Search ─────────────────────────────────────────────────────────
        # Label by what actually ran: HyDE is skipped when no OpenAI key is set
        used_hyde = hypothesis is not None
        retrieval_method = "hyde_dense" if used_hyde else "dense"
        fetch_k = self.dense_prefetch if self.use_reranker else top_k

        sparse_vector, sparse_using = None, None
        if self.use_hybrid:
            t0 = time.time()
            sparse_vector, sparse_using = self._encode_sparse_query(search_text, query_model, domain)
            timings["sparse_encode_ms"] = (time.time() - t0) * 1000
            if sparse_vector is None:
                logger.warning(
                    f"Hybrid requested but no sparse arm for domain '{domain}' — dense only"
                )

        if self.use_hybrid and sparse_vector is not None:
            t0 = time.time()
            search_results = self.vector_store.hybrid_search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=fetch_k,
                dense_prefetch=self.dense_prefetch,
                sparse_prefetch=self.sparse_prefetch,
                document_id=document_id,
                domain=domain,
                sparse_using=sparse_using,
            )
            timings["hybrid_search_ms"] = (time.time() - t0) * 1000
            arm = "bm25" if sparse_using == "bm25" else "sparse"
            retrieval_method = f"hyde_hybrid_{arm}_rrf" if used_hyde else f"hybrid_{arm}_rrf"
        else:
            t0 = time.time()
            search_results = self.vector_store.search(
                query_vector=dense_vector,
                top_k=fetch_k,
                document_id=document_id,
                domain=domain,
            )
            timings["dense_search_ms"] = (time.time() - t0) * 1000

        # ── 4. Rerank ─────────────────────────────────────────────────────────
        if self.use_reranker and self._reranker is not None and search_results:
            t0 = time.time()
            search_results = self._rerank(query, search_results, top_k, retrieval_method)
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
                "total_candidates": self.dense_prefetch + self.sparse_prefetch,
                "final_count": len(search_results),
                "processing_time_ms": round(elapsed_ms, 2),
                "timings": {k: round(v, 2) for k, v in timings.items()},
                "use_hybrid": self.use_hybrid,
                "use_reranker": self.use_reranker,
                "use_hyde": self.use_hyde,
                "hypothesis": hypothesis,
                "reranker_model": self.reranker_model_name if self.use_reranker else None,
                "domain": domain,
                "query_embedding_model": model_name,
            },
        }
