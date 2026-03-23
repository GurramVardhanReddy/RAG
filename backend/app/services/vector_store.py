"""
Vector Store Service

Manages all interactions with the Qdrant vector database:
- Connection management
- Collection creation
- Document storage (with paragraph metadata)
- Similarity search with filtering
- Document deletion
"""

import uuid
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from loguru import logger


class VectorStoreService:
    """
    Manages the Qdrant vector database connection and operations.
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
        """Create the collection if it doesn't exist."""
        if self.client is None:
            logger.warning("Qdrant not connected, skipping collection creation")
            return

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection '{self.collection_name}' ({self.embedding_dimension}d, cosine)")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def add_documents(
        self,
        chunks: list,
        vectors: list[list[float]],
        document_id: str,
        page_paragraphs: dict = None,
    ):
        """
        Store document chunks with their vectors and metadata in Qdrant.

        Args:
            chunks: List of Document objects (from chunking)
            vectors: List of embedding vectors
            document_id: Unique document identifier
            page_paragraphs: Dict mapping page numbers to paragraph lists
                             e.g., {0: [{"paragraph_number": 1, "text": "..."}, ...]}
        """
        if self.client is None:
            logger.error("Qdrant not connected")
            return

        if page_paragraphs is None:
            page_paragraphs = {}

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            page_num = chunk.metadata.get("page", -1)

            # Find which paragraph this chunk belongs to
            para_info = self._find_paragraph_for_chunk(
                chunk.page_content,
                page_paragraphs.get(page_num, [])
            )

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
            f"(with paragraph tracking)"
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

        chunk_clean = chunk_text.strip().lower()
        matching_paras = []

        for para in page_paragraphs:
            para_clean = para["text"].strip().lower()

            # Exact containment
            if chunk_clean in para_clean:
                return {
                    "paragraph_number": para["paragraph_number"],
                    "paragraph_numbers": [para["paragraph_number"]],
                    "match_type": "exact",
                }
            if para_clean in chunk_clean:
                matching_paras.append(para["paragraph_number"])
            elif len(chunk_clean) > 0 and len(para_clean) > 0:
                chunk_start = chunk_clean[:100]
                if chunk_start in para_clean:
                    matching_paras.append(para["paragraph_number"])

        if matching_paras:
            return {
                "paragraph_number": matching_paras[0],
                "paragraph_numbers": matching_paras,
                "match_type": "partial" if len(matching_paras) > 1 else "exact",
            }

        # Best effort: word overlap
        best_para = -1
        best_overlap = 0
        chunk_words = set(chunk_clean.split())
        for para in page_paragraphs:
            para_words = set(para["text"].lower().split())
            overlap = len(chunk_words & para_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_para = para["paragraph_number"]

        return {
            "paragraph_number": best_para,
            "paragraph_numbers": [best_para] if best_para > 0 else [],
            "match_type": "best_effort",
        }

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: str = None,
    ) -> list[dict]:
        """
        Search for similar chunks. Returns results with page AND paragraph numbers.
        """
        if self.client is None:
            logger.error("Qdrant not connected")
            return []

        # Build filter
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
            }
        except Exception:
            return {"status": "connected", "collection": "not_created"}
