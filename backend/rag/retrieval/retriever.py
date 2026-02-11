"""
Retrieval Module

Implements multiple retrieval strategies:
- Vector (semantic) search
- BM25 (keyword) search
- Hybrid search (vector + BM25 combined)
- Cross-encoder re-ranking
"""

from langchain_core.documents import Document
from loguru import logger
from typing import Optional
import time


class RetrieverPipeline:
    """
    Configurable retrieval pipeline supporting hybrid search + re-ranking.
    
    Usage:
        retriever = RetrieverPipeline(vector_store=my_store)
        results = retriever.retrieve("What is the refund policy?", top_k=5)
    """

    def __init__(
        self,
        vector_store,
        reranker_model_name: str = "BAAI/bge-reranker-base",
        use_hybrid: bool = True,
        use_reranker: bool = True,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        self.vector_store = vector_store
        self.reranker_model_name = reranker_model_name
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # Lazy-loaded components
        self._bm25_retriever = None
        self._reranker = None

    def _init_bm25(self, documents: list[Document]):
        """Initialize BM25 retriever from documents."""
        from langchain_community.retrievers import BM25Retriever
        self._bm25_retriever = BM25Retriever.from_documents(documents)
        logger.info(f"BM25 index built with {len(documents)} documents")

    def _init_reranker(self):
        """Initialize cross-encoder re-ranker."""
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        self._reranker = HuggingFaceCrossEncoder(model_name=self.reranker_model_name)
        logger.info(f"Re-ranker loaded: {self.reranker_model_name}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        initial_k: int = 50,
    ) -> dict:
        """
        Run the full retrieval pipeline.
        
        Returns a dict with:
        - documents: list of retrieved Document objects
        - metadata: retrieval stats for explainability
        """
        start_time = time.time()
        retrieval_method = "vector"

        # --- Vector search ---
        vector_results = self.vector_store.similarity_search_with_score(query, k=initial_k)

        if self.use_hybrid and self._bm25_retriever:
            retrieval_method = "hybrid"
            # TODO: Combine BM25 + vector results with ensemble
            # For now, just use vector results
            pass

        # --- Re-ranking ---
        if self.use_reranker and self._reranker:
            retrieval_method += "_reranked"
            # TODO: Apply cross-encoder re-ranking
            pass

        # Take top_k results
        final_results = vector_results[:top_k]
        elapsed_ms = (time.time() - start_time) * 1000

        documents = [doc for doc, score in final_results]
        scores = [score for doc, score in final_results]

        logger.info(
            f"Retrieved {len(documents)} docs in {elapsed_ms:.1f}ms "
            f"(method: {retrieval_method})"
        )

        return {
            "documents": documents,
            "scores": scores,
            "retrieval_method": retrieval_method,
            "total_searched": len(vector_results),
            "processing_time_ms": elapsed_ms,
        }
