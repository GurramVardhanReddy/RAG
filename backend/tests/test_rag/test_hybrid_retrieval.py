"""
Integration tests for domain-aware hybrid retrieval on an in-memory Qdrant.

The dense model is a deterministic fake, so these tests download nothing.
They pin three regressions:
    1. Dense-only domain models silently skipped the sparse arm.
    2. The dense fallback called QdrantClient.search(), removed in
       qdrant-client 1.13, and returned [] for every query.
    3. A collection created before the BM25 arm must degrade to dense-only,
       not crash.
"""

import hashlib

import numpy as np
import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SparseVectorParams, VectorParams

from backend.app.services.vector_store import VectorStoreService
from backend.rag.retrieval.bm25_sparse import BM25SparseEncoder
from backend.rag.retrieval.retriever import HybridRetriever

DOMAIN = "Finance/Regulatory"
MODEL = "fake/dense-model"
DIM = 768


class FakeDenseModel:
    """Deterministic dense embedder: one pseudo-random unit vector per text."""

    def _vec(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).normal(size=DIM)
        return (v / np.linalg.norm(v)).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


DOCS = [
    "UCITS funds must keep at least 10 percent of assets in liquid instruments.",
    "The Basel III capital ratio sets minimum common equity for banks.",
    "AIFMD regulates alternative investment fund managers in the EU.",
    "Quarterly reporting deadlines apply to all regulated entities.",
]


@pytest.fixture
def store():
    vs = VectorStoreService()
    vs.client = QdrantClient(":memory:")
    return vs


def index(vs: VectorStoreService, docs=DOCS, bm25=True):
    chunks = [Document(page_content=t, metadata={"page": 0, "source": "t.pdf"}) for t in docs]
    model = FakeDenseModel()
    vs.add_documents(
        chunks=chunks,
        dense_vectors=model.embed_documents(docs),
        sparse_vectors=[{"indices": [], "values": []}] * len(docs),
        document_id="doc1",
        domain=DOMAIN,
        embedding_model_name=MODEL,
        bm25_vectors=BM25SparseEncoder().embed_documents(docs) if bm25 else None,
    )


def make_retriever(vs: VectorStoreService, use_hybrid=True) -> HybridRetriever:
    r = HybridRetriever(
        vector_store=vs,
        use_hybrid=use_hybrid,
        use_reranker=False,
        dense_prefetch=4,
        sparse_prefetch=4,
        document_registry={"doc1": {"domain": DOMAIN, "embedding_model": MODEL}},
    )
    r._get_query_embedding_model = lambda name: FakeDenseModel()
    return r


class TestHybridRetrieval:

    def test_dense_only_model_gets_bm25_hybrid(self, store):
        index(store)
        out = make_retriever(store).retrieve("AIFMD fund managers", top_k=2, document_id="doc1")
        assert out["metadata"]["retrieval_method"] == "hybrid_bm25_rrf"
        # The fake dense model knows nothing about meaning; only BM25 can find this
        assert "AIFMD" in out["results"][0]["text"]

    def test_dense_fallback_returns_results(self, store):
        index(store)
        out = make_retriever(store, use_hybrid=False).retrieve(
            "anything", top_k=3, document_id="doc1"
        )
        assert out["metadata"]["retrieval_method"] == "dense"
        assert len(out["results"]) == 3

    def test_document_filter(self, store):
        index(store)
        out = make_retriever(store).retrieve("capital ratio", top_k=5, document_id="other-doc")
        assert out["results"] == []

    def test_legacy_collection_degrades_to_dense(self, store):
        # A collection created before the BM25 arm existed has no "bm25" vector
        store.client.create_collection(
            collection_name=store._collection_name(DOMAIN),
            vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        index(store)
        assert store.supports_bm25(DOMAIN) is False
        out = make_retriever(store).retrieve("AIFMD fund managers", top_k=2, document_id="doc1")
        assert out["metadata"]["retrieval_method"] == "dense"
        assert len(out["results"]) == 2
