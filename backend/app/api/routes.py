"""
API route definitions for the Explainable RAG system.
"""

import os
import time
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from backend.app.core.config import get_settings
from backend.app.services.document_loader import load_document
from backend.rag.chunking.chunker import recursive_chunk, semantic_chunk
from backend.rag.generation.generator import generate_answer, build_context
from backend.rag.explainability.engine import ExplainabilityEngine

router = APIRouter()
settings = get_settings()
explainability_engine = ExplainabilityEngine()

# In-memory document registry (tracks uploaded docs)
# In production, this would be a proper database
document_registry: dict = {}


# =====================
# Request/Response Models
# =====================

class QueryRequest(BaseModel):
    """User's question about their uploaded document."""
    question: str
    document_id: Optional[str] = None  # If None, search all documents
    top_k: int = 5


class SourceChunk(BaseModel):
    """A retrieved chunk with explainability metadata."""
    content: str
    page_number: Optional[int] = None
    chunk_index: int
    relevance_score: float


class ExplainabilityInfo(BaseModel):
    """Transparency layer — explains HOW the answer was derived."""
    sources_used: list[SourceChunk]
    confidence_score: float
    retrieval_strategy: str
    total_chunks_searched: int
    processing_time_ms: float


class QueryResponse(BaseModel):
    """Full response with answer + explainability."""
    answer: str
    explainability: ExplainabilityInfo


class UploadResponse(BaseModel):
    """Response after document upload and processing."""
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    message: str


# =====================
# Endpoints
# =====================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF document.
    The system will chunk it, embed it, and store it in Qdrant.
    """
    start_time = time.time()

    # 1. Validate file type
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {settings.allowed_extensions}",
        )

    # 2. Save the uploaded file
    os.makedirs(settings.upload_dir, exist_ok=True)
    document_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(settings.upload_dir, f"{document_id}_{filename}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    logger.info(f"Saved uploaded file: {save_path}")

    # 3. Load the document
    try:
        docs = load_document(save_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load document: {str(e)}")

    total_pages = len(docs)

    # 4. Chunk the document (semantic chunking — groups by meaning)
    embedding_model = request.app.state.embedding_model
    chunks = semantic_chunk(docs, embedding_model=embedding_model)
    total_chunks = len(chunks)

    # 5. Generate embeddings
    #embedding_model = request.app.state.embedding_model
    chunk_texts = [chunk.page_content for chunk in chunks]
    logger.info(f"Generating embeddings for {total_chunks} chunks...")
    vectors = embedding_model.embed_documents(chunk_texts)
    logger.info("Embeddings generated!")

    # 6. Store in Qdrant
    vector_store = request.app.state.vector_store
    if vector_store.client is None:
        raise HTTPException(status_code=503, detail="Vector database is not connected. Is Qdrant running?")

    vector_store.add_documents(chunks, vectors, document_id)

    # 7. Register the document
    elapsed = time.time() - start_time
    document_registry[document_id] = {
        "filename": filename,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "file_path": save_path,
        "processing_time_seconds": round(elapsed, 2),
    }

    logger.info(f"Document '{filename}' processed in {elapsed:.1f}s — {total_chunks} chunks stored")

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        total_pages=total_pages,
        total_chunks=total_chunks,
        message=f"Document processed successfully in {elapsed:.1f}s",
    )


@router.post("/query", response_model=QueryResponse)
async def query_document(request: Request, body: QueryRequest):
    """
    Ask a question about an uploaded document.
    Returns an answer with full explainability metadata.
    """
    start_time = time.time()

    # 1. Get services from app state
    embedding_model = request.app.state.embedding_model
    vector_store = request.app.state.vector_store

    if vector_store.client is None:
        raise HTTPException(status_code=503, detail="Vector database is not connected. Is Qdrant running?")

    # 2. Embed the query
    query_vector = embedding_model.embed_query(body.question)

    # 3. Search Qdrant
    search_results = vector_store.search(
        query_vector=query_vector,
        top_k=body.top_k,
        document_id=body.document_id,
    )

    if not search_results:
        raise HTTPException(status_code=404, detail="No relevant chunks found. Have you uploaded a document?")

    # 4. Convert search results to Documents for the generator
    from langchain_core.documents import Document
    documents = [
        Document(
            page_content=r["text"],
            metadata={"page": r["page_number"], "source": r["source"]},
        )
        for r in search_results
    ]
    scores = [r["score"] for r in search_results]

    # 5. Generate answer with LLM
    try:
        answer = generate_answer(
            question=body.question,
            documents=documents,
            scores=scores,
            api_key=settings.openai_api_key,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")

    # 6. Build explainability info
    elapsed_ms = (time.time() - start_time) * 1000

    # Get total chunks in collection for context
    collection_info = vector_store.get_collection_info()
    total_chunks = collection_info.get("points_count", 0)

    sources_used = [
        SourceChunk(
            content=r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"],
            page_number=r["page_number"],
            chunk_index=r["chunk_index"],
            relevance_score=round(r["score"], 4),
        )
        for r in search_results
    ]

    confidence = explainability_engine.compute_confidence_score(scores, answer, body.question)

    explainability = ExplainabilityInfo(
        sources_used=sources_used,
        confidence_score=round(confidence, 4),
        retrieval_strategy="vector_search_cosine",
        total_chunks_searched=total_chunks,
        processing_time_ms=round(elapsed_ms, 1),
    )

    logger.info(
        f"Query answered in {elapsed_ms:.0f}ms | "
        f"Confidence: {confidence:.2f} | "
        f"Sources: {len(sources_used)}"
    )

    return QueryResponse(answer=answer, explainability=explainability)


@router.get("/documents")
async def list_documents():
    """List all uploaded and processed documents."""
    return {
        "documents": [
            {"document_id": doc_id, **info}
            for doc_id, info in document_registry.items()
        ]
    }


@router.get("/documents/{document_id}")
async def get_document_info(document_id: str):
    """Get metadata about a specific uploaded document."""
    if document_id not in document_registry:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return {"document_id": document_id, **document_registry[document_id]}
