# Retrieval module — hybrid search pipeline
from backend.rag.retrieval.retriever import HybridRetriever
from backend.rag.retrieval.sparse_encoder import encode_sparse, encode_sparse_query

__all__ = ["HybridRetriever", "encode_sparse", "encode_sparse_query"]
