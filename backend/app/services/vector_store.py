"""
Vector Store Service

Manages the Qdrant vector database connection and operations.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_core.documents import Document
from loguru import logger
from typing import Optional
import uuid


class VectorStoreService:
    """
    Manages all interactions with the Qdrant vector database.
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
        self.client = None

    def connect(self):
        """Establish connection to Qdrant."""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            # Test connection
            self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.warning(
                f"Could not connect to Qdrant at {self.host}:{self.port}. "
                f"Error: {e}. Running in offline mode — upload/query will not work until Qdrant is running."
            )
            self.client = None

    def ensure_collection(self):
        """Create the collection if it doesn't exist."""
        if self.client is None:
            raise ConnectionError("Qdrant is not connected. Make sure Qdrant is running.")

        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection '{self.collection_name}' (dim={self.embedding_dimension})")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def add_documents(
        self,
        chunks: list[Document],
        vectors: list[list[float]],
        document_id: str,
    ) -> int:
        """
        Store document chunks with their embeddings in Qdrant.

        Each point stores:
        - vector: the embedding
        - payload: chunk text, page number, document_id, chunk_index
        """
        if self.client is None:
            raise ConnectionError("Qdrant is not connected.")

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk.page_content,
                        "page_number": chunk.metadata.get("page", -1),
                        "source": chunk.metadata.get("source", ""),
                        "document_id": document_id,
                        "chunk_index": i,
                    },
                )
            )

        # Upload in batches of 100
        batch_size = 100
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Stored {len(points)} chunks for document '{document_id}'")
        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for similar chunks.

        Returns list of dicts with: text, page_number, score, document_id, chunk_index
        """
        if self.client is None:
            raise ConnectionError("Qdrant is not connected.")

        # Optional filter: only search within a specific document
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
            raise ConnectionError("Qdrant is not connected.")

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
        """Get info about the current collection."""
        if self.client is None:
            return {"status": "disconnected"}

        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception:
            return {"status": "collection not found"}
