# Retrieval module — hybrid search pipeline
# sparse_encoder (legacy SPLADE, needs fastembed) is not imported here: fastembed is
# not in requirements.txt, and importing it eagerly broke every fresh install.
from backend.rag.retrieval.retriever import HybridRetriever
from backend.rag.retrieval.bm25_sparse import BM25SparseEncoder

__all__ = ["HybridRetriever", "BM25SparseEncoder"]
