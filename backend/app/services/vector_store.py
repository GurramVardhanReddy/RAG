"""
Vector Store Service

Manages all interactions with the Qdrant vector database:
- Connection management
- Collection creation with NAMED VECTORS (dense + sparse)
- Document storage (with paragraph metadata)
- Dense-only search (backward compatible)
- Hybrid search with RRF fusion (dense + sparse)
- Document deletion

Updated for hybrid retrieval:
    The collection now stores two vector types per point:
    - "dense": 768-dim BGE embeddings (cosine similarity)
    - "sparse": SPLADE sparse vectors (dot product)

    Qdrant's native prefetch + RRF fusion runs both searches
    server-side in a single query_points() call.

References:
    - Wang et al., "BERT-based Dense Retrievers Require Interpolation
      with BM25 for Effective Passage Retrieval" (ICTIR 2021)
"""

import uuid
from typing import Optional
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
    NamedVector,
    NamedSparseVector,
)
from loguru import logger


class VectorStoreService:
    """
    Manages the Qdrant vector database connection and operations.
    Supports hybrid retrieval with dense + sparse named vectors.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "explainable_rag",
        embedding_dimension: int = 768,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.client: Optional[QdrantClient] = None

    def connect(self):
        """Establish connection to Qdrant."""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            # Test connection
            self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}")
            self.client = None

    def ensure_collection(self):
        """
        Create the collection with named vectors if it doesn't exist.

        Vector config:
            - "dense": BGE embeddings (768d, cosine)
            - "sparse": SPLADE sparse vectors (dot product, variable dim)
        """
        if self.client is None:
            logger.warning("Qdrant not connected, skipping collection creation")
            return

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )
            logger.info(
                f"Created collection '{self.collection_name}' "
                f"(dense: {self.embedding_dimension}d cosine + sparse: SPLADE)"
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def add_documents(
        self,
        chunks: list,
        dense_vectors: list[list[float]],
        sparse_vectors: list,
        document_id: str,
        page_paragraphs: dict = None,
    ):
        """
        Store document chunks with both dense and sparse vectors.

        Args:
            chunks: List of Document objects (from chunking)
            dense_vectors: List of dense embedding vectors (768d)
            sparse_vectors: List of SparseEmbedding objects from fastembed
                           Each has .indices and .values arrays
            document_id: Unique document identifier
            page_paragraphs: Dict mapping page numbers to paragraph lists
        """
        if self.client is None:
            logger.error("Qdrant not connected")
            return

        if page_paragraphs is None:
            page_paragraphs = {}

        points = []
        for i, (chunk, dense_vec, sparse_vec) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors)
        ):
            page_num = chunk.metadata.get("page", -1)

            # Find which paragraph this chunk belongs to
            para_info = self._find_paragraph_for_chunk(
                chunk.page_content,
                page_paragraphs.get(page_num, [])
            )

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload={
                    "text": chunk.page_content,
                    "page_number": page_num,
                    "paragraph_number": para_info["paragraph_number"],
                    "paragraph_numbers": para_info["paragraph_numbers"],
                    "paragraph_match_type": para_info["match_type"],
                    "source": chunk.metadata.get("source", ""),
                    "document_id": document_id,
                    "chunk_index": i,
                },
            )
            points.append(point)

        # Upload in batches of 100
        batch_size = 100
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(
            f"Stored {len(points)} chunks for document '{document_id}' "
            f"(dense + sparse vectors, with paragraph tracking)"
        )

    def _find_paragraph_for_chunk(
        self, chunk_text: str, page_paragraphs: list[dict]
    ) -> dict:
        """
        Find which paragraph(s) a chunk belongs to based on text overlap.
        """
        if not page_paragraphs:
            return {
                "paragraph_number": -1,
                "paragraph_numbers": [],
                "match_type": "no_paragraphs",
            }

        matched_paragraphs = []
        best_overlap = 0
        best_para_num = -1

        for para in page_paragraphs:
            para_text = para.get("text", "")
            para_num = para.get("paragraph_number", -1)

            if not para_text:
                continue

            # Simple overlap: check if chunk text contains paragraph text or vice versa
            overlap = len(set(chunk_text.split()) & set(para_text.split()))
            if overlap > 0:
                matched_paragraphs.append(para_num)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_para_num = para_num

        if len(matched_paragraphs) == 0:
            return {
                "paragraph_number": -1,
                "paragraph_numbers": [],
                "match_type": "no_match",
            }
        elif len(matched_paragraphs) == 1:
            return {
                "paragraph_number": matched_paragraphs[0],
                "paragraph_numbers": matched_paragraphs,
                "match_type": "exact",
            }
        else:
            return {
                "paragraph_number": best_para_num,
                "paragraph_numbers": matched_paragraphs,
                "match_type": "multi_paragraph",
            }

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: str = None,
    ) -> list[dict]:
        """
        Dense-only search (backward compatible).

        Uses the "dense" named vector for cosine similarity search.
        """
        if self.client is None:
            logger.error("Qdrant not connected")
            return []

        search_filter = None
        if document_id:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )

        return [
            {
                "text": hit.payload.get("text", ""),
                "page_number": hit.payload.get("page_number", -1),
                "paragraph_number": hit.payload.get("paragraph_number", -1),
                "paragraph_numbers": hit.payload.get("paragraph_numbers", []),
                "paragraph_match_type": hit.payload.get("paragraph_match_type", "unknown"),
                "source": hit.payload.get("source", ""),
                "document_id": hit.payload.get("document_id", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "score": hit.score,
                "retrieval_method": "dense",
            }
            for hit in results.points
        ]

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector,
        top_k: int = 5,
        dense_prefetch: int = 40,
        sparse_prefetch: int = 10,
        document_id: str = None,
    ) -> list[dict]:
        """
        Hybrid search using Qdrant's native RRF (Reciprocal Rank Fusion).

        Dense gets more candidates than sparse (40 vs 10 by default) so it
        dominates the RRF candidate pool — effectively weighting dense ~4x
        higher. Tune dense_prefetch/sparse_prefetch in config to adjust.

        Args:
            dense_vector: BGE dense embedding (768d)
            sparse_vector: SparseEmbedding from fastembed (.indices, .values)
            top_k: Number of final results after fusion
            dense_prefetch: Candidates from dense retriever (higher = more weight)
            sparse_prefetch: Candidates from sparse retriever (lower = less weight)
            document_id: Optional filter to search within one document

        Returns:
            List of result dicts with RRF-fused scores
        """
        if self.client is None:
            logger.error("Qdrant not connected")
            return []

        # Build optional document filter
        search_filter = None
        if document_id:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        # Convert fastembed SparseEmbedding to Qdrant SparseVector
        qdrant_sparse = SparseVector(
            indices=sparse_vector.indices.tolist(),
            values=sparse_vector.values.tolist(),
        )

        # Prefetch from both retrievers with asymmetric limits, then fuse with RRF
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=dense_prefetch,
                    filter=search_filter,
                ),
                Prefetch(
                    query=qdrant_sparse,
                    using="sparse",
                    limit=sparse_prefetch,
                    filter=search_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text": hit.payload.get("text", ""),
                "page_number": hit.payload.get("page_number", -1),
                "paragraph_number": hit.payload.get("paragraph_number", -1),
                "paragraph_numbers": hit.payload.get("paragraph_numbers", []),
                "paragraph_match_type": hit.payload.get("paragraph_match_type", "unknown"),
                "source": hit.payload.get("source", ""),
                "document_id": hit.payload.get("document_id", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "score": hit.score,
                "retrieval_method": "hybrid_rrf",
            }
            for hit in results.points
        ]

    def delete_document(self, document_id: str):
        """Delete all chunks belonging to a specific document."""
        if self.client is None:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info(f"Deleted all chunks for document '{document_id}'")

    def get_collection_info(self) -> dict:
        """Get information about the vector collection."""
        if self.client is None:
            return {"status": "disconnected"}
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "status": "connected",
                "collection": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "vector_config": "dense (768d cosine) + sparse (SPLADE)",
            }
        except Exception:
            return {"status": "connected", "collection": "not_created"}
