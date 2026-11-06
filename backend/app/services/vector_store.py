"""
Vector Store Service — Domain-Aware Multi-Collection Qdrant Manager
====================================================================
Each domain uses its own Qdrant collection to ensure correct vector
dimension handling (most domains = 768d, General/bge-m3 = 1024d).

Collection naming convention:
    explainable_rag__finance_regulatory
    explainable_rag__legal
    explainable_rag__medical_biomedical
    explainable_rag__scientific_academic
    explainable_rag__code_software
    explainable_rag__ecommerce_product
    explainable_rag__general

Queries route to the correct collection via document_id → domain lookup
stored in the document registry (routes.py).
"""

import uuid
from loguru import logger
from backend.rag.retrieval.bm25_sparse import BM25SparseEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    Modifier,
)

# ── Domain → collection suffix map ────────────────────────────────────────────
DOMAIN_COLLECTION_SUFFIX: dict[str, str] = {
    "Finance/Regulatory":  "finance_regulatory",
    "Legal":               "legal",
    "Medical/Biomedical":  "medical_biomedical",
    "Scientific/Academic": "scientific_academic",
    "Code/Software":       "code_software",
    "E-commerce/Product":  "ecommerce_product",
    "General":             "general",
}

# Dimensions per domain (must match embedder.py)
DOMAIN_DIMENSIONS: dict[str, int] = {
    "Finance/Regulatory":  768,
    "Legal":               768,
    "Medical/Biomedical":  768,
    "Scientific/Academic": 768,
    "Code/Software":       768,
    "E-commerce/Product":  768,
    "General":             768,
}


class VectorStoreService:
    """
    Manages Qdrant collections for domain-adaptive RAG.

    One collection per domain, each with named vectors:
        - "dense":  Domain-specific embedding (768d, cosine)
        - "sparse": Learned sparse weights (BGE-M3 only; empty for dense-only models)
        - "bm25":   BM25 term weights, IDF applied by Qdrant (every domain)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_prefix: str = "explainable_rag",
        embedding_dimension: int = 768,  # Legacy default; overridden per domain
    ):
        self.host = host
        self.port = port
        self.collection_prefix = collection_prefix
        self.embedding_dimension = embedding_dimension  # kept for backward compat
        self.client: QdrantClient | None = None

        # Track which collections have been created
        self._created_collections: set[str] = set()
        # Collections created before the "bm25" vector existed: dense-only until re-indexed
        self._legacy_collections: set[str] = set()

    # ----------------------------------------------------------
    # CONNECTION
    # ----------------------------------------------------------

    def connect(self):
        """Connect to Qdrant. Non-fatal on failure."""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}")
            self.client = None

    # ----------------------------------------------------------
    # COLLECTION MANAGEMENT
    # ----------------------------------------------------------

    def _collection_name(self, domain: str) -> str:
        """Build the Qdrant collection name for a domain."""
        suffix = DOMAIN_COLLECTION_SUFFIX.get(domain, "general")
        return f"{self.collection_prefix}__{suffix}"

    def ensure_collection_for_domain(self, domain: str):
        """
        Create a domain-specific collection if it doesn't exist.
        Safe to call multiple times — checks existence first.
        """
        if self.client is None:
            logger.warning("Qdrant not connected — skipping collection creation")
            return

        name = self._collection_name(domain)
        dim = DOMAIN_DIMENSIONS.get(domain, 768)

        if name in self._created_collections:
            return  # Already ensured this session

        existing = {c.name for c in self.client.get_collections().collections}
        if name not in existing:
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(size=dim, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                    "bm25": SparseVectorParams(modifier=Modifier.IDF),
                },
            )
            logger.info(
                f"Created collection '{name}' "
                f"(domain={domain}, dense={dim}d cosine + learned sparse + BM25)"
            )
        else:
            sparse_cfg = self.client.get_collection(name).config.params.sparse_vectors or {}
            if "bm25" not in sparse_cfg:
                self._legacy_collections.add(name)
                logger.warning(
                    f"Collection '{name}' predates the BM25 arm: hybrid search is "
                    f"unavailable for it until it is deleted and its documents re-uploaded"
                )
            logger.info(f"Collection '{name}' already exists (domain={domain})")

        self._created_collections.add(name)

    def ensure_collection(self, domain: str = "General"):
        """Backward-compatible alias."""
        self.ensure_collection_for_domain(domain)

    def supports_bm25(self, domain: str) -> bool:
        """True if the domain collection has the BM25 sparse vector."""
        self.ensure_collection_for_domain(domain)
        return self._collection_name(domain) not in self._legacy_collections

    @staticmethod
    def _to_qdrant_sparse(sparse_vec) -> SparseVector:
        """Accept a Qdrant SparseVector, an {"indices", "values"} dict, or a fastembed object."""
        if isinstance(sparse_vec, SparseVector):
            return sparse_vec
        if isinstance(sparse_vec, dict):
            return SparseVector(indices=sparse_vec["indices"], values=sparse_vec["values"])
        return SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist(),
        )

    # ----------------------------------------------------------
    # DOCUMENT STORAGE
    # ----------------------------------------------------------

    def add_documents(
        self,
        chunks: list,
        dense_vectors: list[list[float]],
        sparse_vectors: list,
        document_id: str,
        page_paragraphs: dict = None,
        domain: str = "General",
        embedding_model_name: str = "BAAI/bge-m3",
        bm25_vectors: list | None = None,
    ):
        """
        Store document chunks in the correct domain collection.

        Args:
            chunks: LangChain Document objects from chunker
            dense_vectors: Domain-model dense embeddings
            sparse_vectors: Learned sparse embeddings (empty dicts for dense-only models)
            bm25_vectors: BM25 sparse vectors from BM25SparseEncoder (optional)
            document_id: Unique doc identifier
            page_paragraphs: {page_num: [paragraph_dicts]}
            domain: Detected domain string (e.g. "Finance/Regulatory")
            embedding_model_name: Model used — stored in payload for audit
        """
        if self.client is None:
            logger.error("Qdrant not connected")
            return

        if page_paragraphs is None:
            page_paragraphs = {}

        collection = self._collection_name(domain)
        self.ensure_collection_for_domain(domain)

        if bm25_vectors is None or collection in self._legacy_collections:
            bm25_vectors = [None] * len(chunks)

        points = []
        for i, (chunk, dense_vec, sparse_vec, bm25_vec) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors, bm25_vectors)
        ):
            page_num = chunk.metadata.get("page", -1)
            para_info = self._find_paragraph_for_chunk(
                chunk.page_content,
                page_paragraphs.get(page_num, []),
            )

            vector = {"dense": dense_vec, "sparse": self._to_qdrant_sparse(sparse_vec)}
            if bm25_vec is not None:
                vector["bm25"] = self._to_qdrant_sparse(bm25_vec)

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk.page_content,
                    "page_number": page_num,
                    "paragraph_number": para_info["paragraph_number"],
                    "paragraph_numbers": para_info["paragraph_numbers"],
                    "paragraph_match_type": para_info["match_type"],
                    "source": chunk.metadata.get("source", ""),
                    "document_id": document_id,
                    "chunk_index": i,
                    # Domain-adaptive metadata (NEW)
                    "domain": domain,
                    "embedding_model": embedding_model_name,
                },
            )
            points.append(point)

        # Upload in batches of 100
        for start in range(0, len(points), 100):
            batch = points[start: start + 100]
            self.client.upsert(collection_name=collection, points=batch)

        logger.info(
            f"Stored {len(points)} chunks for document '{document_id}' "
            f"in collection '{collection}' "
            f"(domain={domain}, model={embedding_model_name})"
        )

    # ----------------------------------------------------------
    # RETRIEVAL
    # ----------------------------------------------------------

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector,
        top_k: int = 5,
        dense_prefetch: int = 40,
        sparse_prefetch: int = 10,
        document_id: str = None,
        domain: str = "General",
        sparse_using: str = "sparse",
    ) -> list[dict]:
        """
        Hybrid search (dense + sparse → RRF) in a domain collection.
        Dense gets more candidates (dense_prefetch) than sparse (sparse_prefetch)
        to weight dense results higher. Falls back to dense-only on error.

        Args:
            dense_vector: Domain-model dense embedding
            sparse_vector: Qdrant SparseVector or {"indices", "values"} dict
            sparse_using: Sparse vector name: "sparse" (learned) or "bm25"
            top_k: Final results to return
            dense_prefetch: Candidates from dense arm (default 40)
            sparse_prefetch: Candidates from sparse arm (default 10)
            document_id: Optional document filter
            domain: Domain collection to search
        """
        if self.client is None:
            return []

        collection = self._collection_name(domain)
        doc_filter = None
        if document_id:
            doc_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )

        qdrant_sparse = self._to_qdrant_sparse(sparse_vector)

        try:
            results = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=dense_prefetch,
                        filter=doc_filter,
                    ),
                    Prefetch(
                        query=qdrant_sparse,
                        using=sparse_using,
                        limit=sparse_prefetch,
                        filter=doc_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            return self._format_results(results.points)
        except Exception as e:
            logger.warning(f"Hybrid search failed in '{collection}': {e} — falling back to dense")
            return self.search(dense_vector, top_k, document_id, domain)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: str = None,
        domain: str = "General",
    ) -> list[dict]:
        """Dense-only fallback search."""
        if self.client is None:
            return []

        collection = self._collection_name(domain)
        doc_filter = None
        if document_id:
            doc_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )

        try:
            results = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                using="dense",
                limit=top_k,
                query_filter=doc_filter,
                with_payload=True,
            )
            return self._format_results(results.points)
        except Exception as e:
            logger.error(f"Dense search failed in '{collection}': {e}")
            return []

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _format_results(self, points) -> list[dict]:
        """Convert Qdrant ScoredPoint objects to plain dicts."""
        results = []
        for point in points:
            payload = point.payload or {}
            results.append({
                "text": payload.get("text", ""),
                "score": float(point.score),
                "page_number": payload.get("page_number", -1),
                "paragraph_number": payload.get("paragraph_number", -1),
                "paragraph_numbers": payload.get("paragraph_numbers", []),
                "paragraph_match_type": payload.get("paragraph_match_type", "unknown"),
                "source": payload.get("source", ""),
                "document_id": payload.get("document_id", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "content_type": payload.get("content_type", "text"),
                "domain": payload.get("domain", "General"),
                "embedding_model": payload.get("embedding_model", "unknown"),
            })
        return results

    def _find_paragraph_for_chunk(
        self, chunk_text: str, page_paragraphs: list[dict]
    ) -> dict:
        """Find which paragraph(s) a chunk belongs to based on text overlap."""
        if not page_paragraphs:
            return {"paragraph_number": -1, "paragraph_numbers": [], "match_type": "no_paragraphs"}

        best_match = None
        best_overlap = 0.0
        all_matches = []

        chunk_words = set(chunk_text.lower().split())
        if not chunk_words:
            return {"paragraph_number": -1, "paragraph_numbers": [], "match_type": "empty_chunk"}

        for para in page_paragraphs:
            para_text = para.get("text", "")
            para_words = set(para_text.lower().split())
            if not para_words:
                continue
            intersection = len(chunk_words & para_words)
            overlap = intersection / len(chunk_words)
            if overlap > 0.3:
                all_matches.append(para.get("paragraph_number", -1))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = para

        if best_match and best_overlap > 0.3:
            return {
                "paragraph_number": best_match.get("paragraph_number", -1),
                "paragraph_numbers": sorted(set(all_matches)),
                "match_type": "overlap",
            }

        return {"paragraph_number": -1, "paragraph_numbers": [], "match_type": "no_match"}

    def get_collection_info(self, domain: str = "General") -> dict:
        """Return collection stats for a given domain."""
        if self.client is None:
            return {"status": "disconnected"}
        try:
            name = self._collection_name(domain)
            info = self.client.get_collection(name)
            return {
                "status": "connected",
                "collection": name,
                "domain": domain,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except Exception:
            return {"status": "connected", "collection": "not_created_yet", "domain": domain}

    def get_all_collection_info(self) -> list[dict]:
        """Return stats for all domain collections."""
        if self.client is None:
            return [{"status": "disconnected"}]
        results = []
        for domain in DOMAIN_COLLECTION_SUFFIX:
            results.append(self.get_collection_info(domain))
        return results

    def add_visual_documents(
        self,
        visual_docs: list,
        dense_vectors: list[list[float]],
        sparse_vectors: list,
        document_id: str,
        domain: str = "General",
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
    ):
        """
        Store visual element chunks (tables/images) in the correct domain collection.
        Mirrors add_documents() but with visual-specific payload fields.
        """
        if self.client is None:
            logger.error("Qdrant not connected")
            return

        collection = self._collection_name(domain)
        self.ensure_collection_for_domain(domain)

        points = []
        bm25 = BM25SparseEncoder() if collection not in self._legacy_collections else None
        for i, (doc, dense_vec) in enumerate(zip(visual_docs, dense_vectors)):
            vector_data = {"dense": dense_vec}
            if bm25 is not None:
                vector_data["bm25"] = self._to_qdrant_sparse(bm25.embed_document(doc.page_content))

            if sparse_vectors and i < len(sparse_vectors):
                sv = sparse_vectors[i]
                if isinstance(sv, dict):
                    vector_data["sparse"] = SparseVector(
                        indices=sv.get("indices", []),
                        values=sv.get("values", []),
                    )
                else:
                    vector_data["sparse"] = SparseVector(
                        indices=sv.indices.tolist(),
                        values=sv.values.tolist(),
                    )

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector_data,
                payload={
                    "text": doc.page_content,
                    "page_number": doc.metadata.get("page", -1),
                    "paragraph_number": doc.metadata.get("paragraph_number", -1),
                    "paragraph_numbers": [],
                    "paragraph_match_type": "visual_element",
                    "content_type": doc.metadata.get("content_type", "visual"),
                    "table_number": doc.metadata.get("table_number", -1),
                    "figure_number": doc.metadata.get("figure_number", -1),
                    "source": doc.metadata.get("source", ""),
                    "document_id": document_id,
                    "chunk_index": -(i + 1),
                    "image_path": doc.metadata.get("image_path", ""),
                    "domain": domain,
                    "embedding_model": embedding_model_name,
                },
            )
            points.append(point)

        for start in range(0, len(points), 100):
            batch = points[start: start + 100]
            self.client.upsert(collection_name=collection, points=batch)

        n_tables = sum(1 for d in visual_docs if d.metadata.get("content_type") == "table")
        n_images = sum(1 for d in visual_docs if d.metadata.get("content_type") == "image")
        logger.info(
            f"Stored {len(points)} visual chunks for document '{document_id}' "
            f"in collection '{collection}' ({n_tables} tables, {n_images} images)"
        )