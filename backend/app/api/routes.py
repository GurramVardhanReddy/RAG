"""
API route definitions for the Explainable RAG system.

Integrates:
- Document upload with paragraph detection
- Query with page + paragraph attribution
- Full metrics tracking per query
"""

import os
import uuid
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from backend.app.services.document_loader import load_document
from backend.app.services.vector_store import VectorStoreService
from backend.app.core.config import get_settings
from backend.rag.chunking.chunker import recursive_chunk, semantic_chunk
from backend.rag.generation.generator import generate_answer, build_context
from backend.rag.explainability.engine import ExplainabilityEngine
from backend.rag.metrics.tracker import RAGMetricsTracker, ResponseQualityMetrics, ResourceMetrics

router = APIRouter()
settings = get_settings()

# In-memory document registry
document_registry = {}

# Global metrics tracker — persists across requests
metrics_tracker = RAGMetricsTracker(output_dir="./metrics_output")

# Explainability engine
explainability_engine = ExplainabilityEngine()


# =====================
# Request/Response Models
# =====================

class QueryRequest(BaseModel):
    """User's question about their uploaded document."""
    question: str
    document_id: str
    top_k: int = 5


class SourceChunk(BaseModel):
    """A retrieved chunk with explainability metadata including paragraph info."""
    content: str
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    paragraph_numbers: list[int] = []
    paragraph_match_type: str = "unknown"
    chunk_index: int = 0
    relevance_score: float = 0.0
    retrieval_method: str = "vector_search_cosine"


class ExplainabilityInfo(BaseModel):
    """Transparency layer — explains HOW the answer was derived."""
    sources_used: list[SourceChunk]
    confidence_score: float
    retrieval_strategy: str
    total_chunks_searched: int
    processing_time_ms: float


class MetricsInfo(BaseModel):
    """Per-query metrics (subset exposed in API response)."""
    query_to_vector_ms: float = 0.0
    similarity_search_ms: float = 0.0
    generation_ms: float = 0.0
    explainability_ms: float = 0.0
    end_to_end_ms: float = 0.0
    memory_used_mb: float = 0.0
    token_count_prompt: int = 0
    faithfulness_score: float = 0.0
    relevancy_score: float = 0.0


class QueryResponse(BaseModel):
    """Full response with answer + explainability + metrics."""
    answer: str
    explainability: ExplainabilityInfo
    metrics: Optional[MetricsInfo] = None


class UploadResponse(BaseModel):
    """Response after document upload and processing."""
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    total_paragraphs: int
    processing_time_ms: float
    message: str


# =====================
# Endpoints
# =====================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF/DOCX document.
    Chunks it, embeds it, stores in vector DB with paragraph metadata.
    """
    start_time = time.time()

    # 1. Validate file type
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not supported. Use .pdf, .docx, or .txt")

    # 2. Save to disk
    document_id = str(uuid.uuid4())[:8]
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, f"{document_id}_{filename}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    logger.info(f"Saved {filename} as {save_path}")

    # 3. Load document (with paragraph detection)
    docs = load_document(save_path)

    # Build page_paragraphs map for vector store
    page_paragraphs = {}
    total_paragraphs = 0
    for doc in docs:
        page_num = doc.metadata.get("page", 0)
        paragraphs = doc.metadata.get("paragraphs", [])
        page_paragraphs[page_num] = paragraphs
        total_paragraphs += len(paragraphs)

    # 4. Chunk the document (semantic chunking)
    embedding_model = request.app.state.embedding_model
    chunks = semantic_chunk(docs, embedding_model=embedding_model)

    # 5. Generate embeddings
    chunk_texts = [chunk.page_content for chunk in chunks]
    vectors = embedding_model.embed_documents(chunk_texts)

    # 6. Store in Qdrant (with paragraph metadata)
    vector_store: VectorStoreService = request.app.state.vector_store
    vector_store.add_documents(
        chunks, vectors, document_id,
        page_paragraphs=page_paragraphs,
    )

    # 7. Register document
    processing_time = (time.time() - start_time) * 1000
    document_registry[document_id] = {
        "filename": filename,
        "total_pages": len(docs),
        "total_chunks": len(chunks),
        "total_paragraphs": total_paragraphs,
        "processing_time_ms": processing_time,
    }

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        total_pages=len(docs),
        total_chunks=len(chunks),
        total_paragraphs=total_paragraphs,
        processing_time_ms=round(processing_time, 2),
        message=f"Document processed: {len(docs)} pages, {len(chunks)} chunks, {total_paragraphs} paragraphs detected",
    )


@router.post("/query", response_model=QueryResponse)
async def query_document(request: Request, body: QueryRequest):
    """
    Ask a question about an uploaded document.
    Returns answer with page/paragraph attribution and full metrics.
    """
    # --- Start metrics tracking ---
    metrics_record = metrics_tracker.start_query(body.question)

    vector_store: VectorStoreService = request.app.state.vector_store
    embedding_model = request.app.state.embedding_model

    # --- A1. Embed the query ---
    with metrics_tracker.timer.time_stage("query_to_vector"):
        query_vector = embedding_model.embed_query(body.question)

    # --- A2. Similarity search ---
    with metrics_tracker.timer.time_stage("similarity_search"):
        search_results = vector_store.search(
            query_vector=query_vector,
            top_k=body.top_k,
            document_id=body.document_id,
        )

    if not search_results:
        raise HTTPException(
            status_code=404,
            detail="No matching chunks found. Upload a document first."
        )

    # --- A3. Generate answer ---
    with metrics_tracker.timer.time_stage("generation"):
        answer = generate_answer(
            question=body.question,
            search_results=search_results,
            model_name="gpt-4o",
            temperature=0,
            api_key=settings.openai_api_key,
        )

    # --- A4. Build explainability ---
    with metrics_tracker.timer.time_stage("explainability_generation"):
        scores = [r["score"] for r in search_results]
        confidence = explainability_engine.compute_confidence_score(
            scores, answer, body.question
        )

        # Get collection info for total chunks
        collection_info = vector_store.get_collection_info()
        total_chunks = collection_info.get("points_count", 0)

        # Build source chunks with paragraph info
        sources = [
            SourceChunk(
                content=r["text"][:500],  # Truncate for response size
                page_number=r.get("page_number", -1),
                paragraph_number=r.get("paragraph_number", -1),
                paragraph_numbers=r.get("paragraph_numbers", []),
                paragraph_match_type=r.get("paragraph_match_type", "unknown"),
                chunk_index=r.get("chunk_index", 0),
                relevance_score=round(r.get("score", 0.0), 4),
                retrieval_method="vector_search_cosine",
            )
            for r in search_results
        ]

    # --- Compute response quality metrics (fallback/heuristic) ---
    chunk_texts = [r["text"] for r in search_results]
    relevancy = ResponseQualityMetrics.relevancy_score(
        body.question, answer, chunk_texts
    )
    faithfulness = ResponseQualityMetrics.faithfulness_score(
        answer, chunk_texts
    )

    # --- Compute resource metrics ---
    context_usage = ResourceMetrics.compute_context_usage(chunk_texts)

    # --- Populate metrics record ---
    metrics_record.chunks_retrieved = len(search_results)
    metrics_record.chunks_used_in_prompt = len(search_results)
    metrics_record.token_count_prompt = context_usage["total_tokens_in_prompt"]
    metrics_record.relevancy_score = relevancy
    metrics_record.faithfulness_score = faithfulness

    # --- Finish metrics tracking ---
    metrics_record = metrics_tracker.finish_query(metrics_record)

    # --- Calculate end-to-end time for response ---
    end_to_end_ms = metrics_record.end_to_end_time * 1000

    # --- Build response ---
    return QueryResponse(
        answer=answer,
        explainability=ExplainabilityInfo(
            sources_used=sources,
            confidence_score=round(confidence, 4),
            retrieval_strategy="vector_search_cosine",
            total_chunks_searched=total_chunks,
            processing_time_ms=round(end_to_end_ms, 2),
        ),
        metrics=MetricsInfo(
            query_to_vector_ms=round((metrics_record.query_to_vector_time) * 1000, 2),
            similarity_search_ms=round((metrics_record.similarity_search_time) * 1000, 2),
            generation_ms=round((metrics_record.generation_time) * 1000, 2),
            explainability_ms=round((metrics_record.explainability_generation_time) * 1000, 2),
            end_to_end_ms=round(end_to_end_ms, 2),
            memory_used_mb=round(metrics_record.memory_used_mb, 2),
            token_count_prompt=metrics_record.token_count_prompt,
            faithfulness_score=round(faithfulness, 4),
            relevancy_score=round(relevancy, 4),
        ),
    )


@router.get("/documents")
async def list_documents():
    """List all uploaded and processed documents."""
    return {"documents": document_registry}


@router.get("/metrics")
async def get_metrics():
    """
    Get aggregated metrics across all queries.
    Also exports to JSON file for analysis.
    """
    # Export to file
    filepath = metrics_tracker.export_results()
    csv_path = metrics_tracker.export_csv()

    return {
        "total_queries": len(metrics_tracker.records),
        "summary": metrics_tracker._compute_summary(),
        "exported_to": filepath,
        "csv_exported_to": csv_path,
        "records": [r.to_dict() for r in metrics_tracker.records[-10:]],  # Last 10
    }


@router.get("/metrics/export")
async def export_metrics():
    """Export all metrics to JSON and CSV files."""
    json_path = metrics_tracker.export_results()
    csv_path = metrics_tracker.export_csv()
    return {
        "json_file": json_path,
        "csv_file": csv_path,
        "total_records": len(metrics_tracker.records),
    }
