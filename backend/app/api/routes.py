"""
API route definitions for the Explainable RAG system.

Updated with THREE-LAYER EXPLAINABILITY FRAMEWORK:
- Layer 1: Quantitative metrics (attribution, provenance, confidence, faithfulness)
- Layer 2: Visualization data (heatmap, provenance trace)
- Layer 3: Expert evaluation template

Every query now returns full explainability analysis alongside the answer.
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
from backend.rag.retrieval.sparse_encoder import encode_sparse
from backend.rag.metrics.tracker import RAGMetricsTracker, ResponseQualityMetrics, ResourceMetrics
from backend.rag.multimodal.processor import extract_and_embed_visuals

router = APIRouter()
settings = get_settings()

# In-memory document registry
document_registry = {}

# Global metrics tracker — persists across requests
metrics_tracker = RAGMetricsTracker(output_dir="./metrics_output")

# Explainability engine (three-layer)
explainability_engine = ExplainabilityEngine()


# =====================================================================
# REQUEST / RESPONSE MODELS
# =====================================================================

class QueryRequest(BaseModel):
    """User's question about their uploaded document."""
    question: str
    document_id: str
    top_k: int = 5
    run_perturbation_test: bool = False  # Optional: expensive faithfulness test
    run_lime_analysis: bool = False      # Optional: LIME on retrieval queries


# --- Source Chunk (unchanged) ---
class SourceChunk(BaseModel):
    """A retrieved chunk with explainability metadata."""
    content: str
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    paragraph_numbers: list[int] = []
    paragraph_match_type: str = "unknown"
    chunk_index: int = 0
    relevance_score: float = 0.0
    retrieval_method: str = "hybrid_rrf_reranked"


# --- Layer 1: Quantitative Explainability Metrics ---
class SourceAttributionResponse(BaseModel):
    """How accurately does the system attribute answers to sources?"""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    cited_chunk_count: int = 0
    relevant_chunk_count: int = 0
    correctly_cited: int = 0


class ProvenanceCompletenessResponse(BaseModel):
    """How traceable is the answer back to specific sources?"""
    completeness_score: float = 0.0
    total_claims: int = 0
    claims_with_provenance: int = 0
    claims_without_provenance: int = 0
    granularity_score: float = 0.0
    claim_details: list[dict] = []


class ConfidenceCalibrationResponse(BaseModel):
    """How well does confidence predict actual correctness?"""
    confidence_score: float = 0.0
    calibration_bucket: str = ""
    confidence_tier: str = ""
    score_components: dict = {}


class ExplanationFaithfulnessResponse(BaseModel):
    """Are the cited sources actually responsible for the answer?"""
    overall_faithfulness: float = 0.0
    chunk_faithfulness: list[dict] = []
    perturbation_tested: bool = False
    answer_grounding_ratio: float = 0.0


class QuantitativeMetrics(BaseModel):
    """Layer 1: All quantitative explainability metrics."""
    source_attribution: SourceAttributionResponse
    provenance_completeness: ProvenanceCompletenessResponse
    confidence_calibration: ConfidenceCalibrationResponse
    explanation_faithfulness: ExplanationFaithfulnessResponse


# --- Layer 2: Visualization Data ---
class ChunkRelevanceHeatmapResponse(BaseModel):
    """Data for rendering a chunk relevance heatmap (RAG saliency map)."""
    query: str = ""
    chunks: list[dict] = []
    scores: list[float] = []
    score_distribution: dict = {}
    max_score: float = 0.0
    min_score: float = 0.0
    mean_score: float = 0.0
    score_spread: float = 0.0


class ProvenanceTraceResponse(BaseModel):
    """Full audit trail from query to answer."""
    query: str = ""
    retrieval_method: str = ""
    retrieval_stages: list[dict] = []
    chunk_to_answer_mapping: list[dict] = []
    provenance_chain: list[dict] = []


class VisualizationData(BaseModel):
    """Layer 2: All visualization data for the frontend."""
    chunk_relevance_heatmap: ChunkRelevanceHeatmapResponse
    provenance_trace: ProvenanceTraceResponse


# --- Layer 3: Expert Evaluation Template ---
class ExpertEvaluationResponse(BaseModel):
    """Template for domain expert to rate explanation quality."""
    query: str = ""
    answer: str = ""
    source_citations: list[dict] = []
    evaluation_dimensions: dict = {}


# --- LIME Analysis ---
class LimeAnalysisResponse(BaseModel):
    """LIME on retrieval queries — which words drove retrieval?"""
    word_importance: list[dict] = []
    query_sensitivity: float = 0.0
    most_important_words: list[dict] = []
    least_important_words: list[dict] = []
    stability_score: float = 0.0
    total_perturbations: int = 0
    original_query: str = ""


# --- Combined Three-Layer Explainability ---
class ThreeLayerExplainabilityResponse(BaseModel):
    """
    NOVEL RESEARCH CONTRIBUTION:
    Three-layer explainability framework for RAG systems.
    No existing framework (RAGAS, eRAG, GaRAGe) provides this.
    """
    # Summary scores for quick overview
    summary: dict = {}

    # Layer 1: Quantitative Metrics (Objective)
    layer1_quantitative: QuantitativeMetrics

    # Layer 2: Visualization Data (Structural)
    layer2_visualization: VisualizationData

    # Layer 3: Expert Evaluation (Subjective)
    layer3_expert_evaluation: ExpertEvaluationResponse


# --- Legacy Explainability (kept for backward compatibility) ---
class ExplainabilityInfo(BaseModel):
    """Original transparency layer — kept for backward compatibility."""
    sources_used: list[SourceChunk]
    confidence_score: float
    retrieval_strategy: str
    total_chunks_searched: int
    processing_time_ms: float
    retrieval_timings: Optional[dict] = None


# --- Pipeline Metrics ---
class MetricsInfo(BaseModel):
    """Per-query pipeline performance metrics."""
    query_to_vector_ms: float = 0.0
    similarity_search_ms: float = 0.0
    generation_ms: float = 0.0
    explainability_ms: float = 0.0
    end_to_end_ms: float = 0.0
    memory_used_mb: float = 0.0
    token_count_prompt: int = 0
    faithfulness_score: float = 0.0
    relevancy_score: float = 0.0


# --- Full Query Response ---
class QueryResponse(BaseModel):
    """
    Full response with answer + three-layer explainability + metrics.

    This is the primary API output demonstrating the research contribution.
    """
    answer: str

    # Three-layer explainability (NOVEL)
    three_layer_explainability: ThreeLayerExplainabilityResponse

    # LIME analysis (NOVEL — XAI technique on retrieval)
    lime_analysis: Optional[LimeAnalysisResponse] = None

    # Legacy explainability (backward compatible)
    explainability: ExplainabilityInfo

    # Pipeline metrics
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
    visual_tables: int = 0
    visual_images: int = 0
    visual_chunks: int = 0


# =====================================================================
# ENDPOINTS
# =====================================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF/DOCX document.
    Chunks it, generates DENSE + SPARSE embeddings, stores in Qdrant
    with paragraph metadata. For PDFs, also extracts tables/images
    and stores their GPT-4o summaries as additional searchable chunks.
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

    # 5. Generate DENSE embeddings (BGE)
    chunk_texts = [chunk.page_content for chunk in chunks]
    dense_vectors = embedding_model.embed_documents(chunk_texts)
    logger.info(f"Generated {len(dense_vectors)} dense vectors")

    # 6. Generate SPARSE embeddings (SPLADE)
    sparse_vectors = encode_sparse(chunk_texts, model_name=settings.sparse_model)
    logger.info(f"Generated {len(sparse_vectors)} sparse vectors")

    # 7. Store in Qdrant with both vector types + paragraph metadata
    vector_store: VectorStoreService = request.app.state.vector_store
    vector_store: VectorStoreService = request.app.state.vector_store
    vector_store.add_documents(
        chunks=chunks,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        document_id=document_id,
        page_paragraphs=page_paragraphs,
    )

    # 8. Extract and embed visual elements (tables, images) for PDFs
    visual_stats = {"tables": 0, "images": 0, "chunks_added": 0}
    if ext == ".pdf":
        try:
            visual_stats = extract_and_embed_visuals(
                file_path=save_path,
                document_id=document_id,
                openai_api_key=settings.openai_api_key,
                embedding_model=embedding_model,
                vector_store=vector_store,
                sparse_encoder=None,
                output_dir="data/extracted_visuals",
            )
        except Exception as e:
            logger.warning(f"Multimodal extraction failed (non-fatal): {e}")

    # 9. Register document
    processing_time = (time.time() - start_time) * 1000
    document_registry[document_id] = {
        "filename": filename,
        "total_pages": len(docs),
        "total_chunks": len(chunks),
        "total_paragraphs": total_paragraphs,
        "processing_time_ms": processing_time,
        "vector_types": ["dense (BGE)", "sparse (SPLADE)"],
        "visual_elements": visual_stats,
    }

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        total_pages=len(docs),
        total_chunks=len(chunks) + visual_stats["chunks_added"],
        total_paragraphs=total_paragraphs,
        processing_time_ms=round(processing_time, 2),
        message=(
            f"Document processed: {len(docs)} pages, {len(chunks)} text chunks, "
            f"{visual_stats['tables']} tables, {visual_stats['images']} images"
        ),
        visual_tables=visual_stats["tables"],
        visual_images=visual_stats["images"],
        visual_chunks=visual_stats["chunks_added"],
    )


@router.post("/query", response_model=QueryResponse)
async def query_document(request: Request, body: QueryRequest):
    """
    Ask a question about an uploaded document.

    Pipeline:
        Query → Dense + Sparse → RRF Fusion → Rerank → GPT-4o → Answer
        → Three-Layer Explainability Analysis

    Returns answer with full three-layer explainability and pipeline metrics.
    """
    # --- Start metrics tracking ---
    metrics_record = metrics_tracker.start_query(body.question)

    vector_store: VectorStoreService = request.app.state.vector_store
    retriever = request.app.state.retriever

    # =========================================================
    # STAGE 1: HYBRID RETRIEVAL (Dense + Sparse → RRF → Rerank)
    # =========================================================
    with metrics_tracker.timer.time_stage("query_to_vector"):
        pass  # Encoding is done inside retriever.retrieve()

    with metrics_tracker.timer.time_stage("similarity_search"):
        retrieval_output = retriever.retrieve(
            query=body.question,
            top_k=body.top_k,
            document_id=body.document_id,
        )

    search_results = retrieval_output["results"]
    retrieval_metadata = retrieval_output["metadata"]

    if not search_results:
        raise HTTPException(
            status_code=404,
            detail="No matching chunks found. Upload a document first."
        )

    # =========================================================
    # STAGE 2: GENERATION (GPT-4o)
    # =========================================================
    with metrics_tracker.timer.time_stage("generation"):
        answer = generate_answer(
            question=body.question,
            search_results=search_results,
            model_name="gpt-4o",
            temperature=0,
            api_key=settings.openai_api_key,
        )

    # =========================================================
    # STAGE 3: THREE-LAYER EXPLAINABILITY ANALYSIS (NOVEL)
    # =========================================================
    with metrics_tracker.timer.time_stage("explainability_generation"):

        # Get collection info for total chunks count
        collection_info = vector_store.get_collection_info()
        total_chunks = collection_info.get("points_count", 0)

        # Run full three-layer analysis
        three_layer = explainability_engine.analyze(
            question=body.question,
            answer=answer,
            search_results=search_results,
            retrieval_metadata=retrieval_metadata,
            total_chunks=total_chunks,
        )

        # Optional: Run perturbation test (expensive — N extra LLM calls)
        perturbation_results = None
        if body.run_perturbation_test:
            logger.info("Running perturbation-based faithfulness test...")

            def _generate_fn(q, results):
                return generate_answer(
                    question=q,
                    search_results=results,
                    model_name="gpt-4o",
                    temperature=0,
                    api_key=settings.openai_api_key,
                )

            perturbation_results = explainability_engine.run_perturbation_test(
                question=body.question,
                search_results=search_results,
                original_answer=answer,
                generate_fn=_generate_fn,
            )
            # Update faithfulness metrics with perturbation results
            if perturbation_results:
                faithful_count = sum(
                    1 for r in perturbation_results if r.get("is_faithful")
                )
                three_layer.explanation_faithfulness.perturbation_tested = True
                three_layer.explanation_faithfulness.overall_faithfulness = (
                    faithful_count / len(perturbation_results)
                    if perturbation_results else 0.0
                )
                three_layer.explanation_faithfulness.chunk_faithfulness = perturbation_results

        # Optional: Run LIME on retrieval queries (lightweight — no LLM calls)
        lime_result = None
        if body.run_lime_analysis:
            logger.info("Running LIME analysis on retrieval query...")

            def _retrieve_fn(q, k):
                """Lightweight retrieval function for LIME perturbations."""
                output = retriever.retrieve(
                    query=q,
                    top_k=k,
                    document_id=body.document_id,
                )
                return output["results"]

            lime_result = explainability_engine.run_lime_on_query(
                query=body.question,
                original_results=search_results,
                retrieve_fn=_retrieve_fn,
                top_k=body.top_k,
            )

        # Build legacy source chunks (backward compatibility)
        sources = [
            SourceChunk(
                content=r["text"][:500],
                page_number=r.get("page_number", -1),
                paragraph_number=r.get("paragraph_number", -1),
                paragraph_numbers=r.get("paragraph_numbers", []),
                paragraph_match_type=r.get("paragraph_match_type", "unknown"),
                chunk_index=r.get("chunk_index", 0),
                relevance_score=round(r.get("score", 0.0), 4),
                retrieval_method=r.get("retrieval_method", retrieval_metadata["retrieval_method"]),
            )
            for r in search_results
        ]

    # =========================================================
    # STAGE 4: PIPELINE QUALITY METRICS
    # =========================================================
    chunk_texts = [r["text"] for r in search_results]
    relevancy = ResponseQualityMetrics.relevancy_score(
        body.question, answer, chunk_texts
    )
    faithfulness = ResponseQualityMetrics.faithfulness_score(
        answer, chunk_texts
    )

    # Resource metrics
    context_usage = ResourceMetrics.compute_context_usage(chunk_texts)

    # Populate metrics record
    metrics_record.chunks_retrieved = len(search_results)
    metrics_record.chunks_used_in_prompt = len(search_results)
    metrics_record.token_count_prompt = context_usage["total_tokens_in_prompt"]
    metrics_record.relevancy_score = relevancy
    metrics_record.faithfulness_score = faithfulness

    # Finish metrics tracking
    metrics_record = metrics_tracker.finish_query(metrics_record)
    end_to_end_ms = metrics_record.end_to_end_time * 1000

    # =========================================================
    # BUILD RESPONSE
    # =========================================================

    # --- Three-Layer Explainability Response ---
    three_layer_response = ThreeLayerExplainabilityResponse(
        summary=three_layer.summary(),

        layer1_quantitative=QuantitativeMetrics(
            source_attribution=SourceAttributionResponse(
                precision=round(three_layer.source_attribution.precision, 4),
                recall=round(three_layer.source_attribution.recall, 4),
                f1_score=round(three_layer.source_attribution.f1_score, 4),
                cited_chunk_count=three_layer.source_attribution.cited_chunk_count,
                relevant_chunk_count=three_layer.source_attribution.relevant_chunk_count,
                correctly_cited=three_layer.source_attribution.correctly_cited,
            ),
            provenance_completeness=ProvenanceCompletenessResponse(
                completeness_score=round(three_layer.provenance_completeness.completeness_score, 4),
                total_claims=three_layer.provenance_completeness.total_claims,
                claims_with_provenance=three_layer.provenance_completeness.claims_with_provenance,
                claims_without_provenance=three_layer.provenance_completeness.claims_without_provenance,
                granularity_score=round(three_layer.provenance_completeness.granularity_score, 4),
                claim_details=three_layer.provenance_completeness.claim_details,
            ),
            confidence_calibration=ConfidenceCalibrationResponse(
                confidence_score=round(three_layer.confidence_calibration.confidence_score, 4),
                calibration_bucket=three_layer.confidence_calibration.calibration_bucket,
                confidence_tier=three_layer.confidence_calibration.confidence_tier,
                score_components=three_layer.confidence_calibration.score_components,
            ),
            explanation_faithfulness=ExplanationFaithfulnessResponse(
                overall_faithfulness=round(three_layer.explanation_faithfulness.overall_faithfulness, 4),
                chunk_faithfulness=three_layer.explanation_faithfulness.chunk_faithfulness,
                perturbation_tested=three_layer.explanation_faithfulness.perturbation_tested,
                answer_grounding_ratio=round(three_layer.explanation_faithfulness.answer_grounding_ratio, 4),
            ),
        ),

        layer2_visualization=VisualizationData(
            chunk_relevance_heatmap=ChunkRelevanceHeatmapResponse(
                query=three_layer.chunk_relevance_heatmap.query,
                chunks=three_layer.chunk_relevance_heatmap.chunks,
                scores=three_layer.chunk_relevance_heatmap.scores,
                score_distribution=three_layer.chunk_relevance_heatmap.score_distribution,
                max_score=three_layer.chunk_relevance_heatmap.max_score,
                min_score=three_layer.chunk_relevance_heatmap.min_score,
                mean_score=three_layer.chunk_relevance_heatmap.mean_score,
                score_spread=three_layer.chunk_relevance_heatmap.score_spread,
            ),
            provenance_trace=ProvenanceTraceResponse(
                query=three_layer.provenance_trace.query,
                retrieval_method=three_layer.provenance_trace.retrieval_method,
                retrieval_stages=three_layer.provenance_trace.retrieval_stages,
                chunk_to_answer_mapping=three_layer.provenance_trace.chunk_to_answer_mapping,
                provenance_chain=three_layer.provenance_trace.provenance_chain,
            ),
        ),

        layer3_expert_evaluation=ExpertEvaluationResponse(
            query=three_layer.expert_evaluation.query,
            answer=three_layer.expert_evaluation.answer,
            source_citations=three_layer.expert_evaluation.source_citations,
            evaluation_dimensions=three_layer.expert_evaluation.evaluation_dimensions,
        ),
    )

    # --- Legacy Explainability (backward compatible) ---
    legacy_explainability = ExplainabilityInfo(
        sources_used=sources,
        confidence_score=round(three_layer.confidence_calibration.confidence_score, 4),
        retrieval_strategy=retrieval_metadata["retrieval_method"],
        total_chunks_searched=total_chunks,
        processing_time_ms=round(end_to_end_ms, 2),
        retrieval_timings=retrieval_metadata.get("timings", {}),
    )

    # --- Pipeline Metrics ---
    pipeline_metrics = MetricsInfo(
        query_to_vector_ms=round((metrics_record.query_to_vector_time) * 1000, 2),
        similarity_search_ms=round((metrics_record.similarity_search_time) * 1000, 2),
        generation_ms=round((metrics_record.generation_time) * 1000, 2),
        explainability_ms=round((metrics_record.explainability_generation_time) * 1000, 2),
        end_to_end_ms=round(end_to_end_ms, 2),
        memory_used_mb=round(metrics_record.memory_used_mb, 2),
        token_count_prompt=metrics_record.token_count_prompt,
        faithfulness_score=round(faithfulness, 4),
        relevancy_score=round(relevancy, 4),
    )

    return QueryResponse(
        answer=answer,
        three_layer_explainability=three_layer_response,
        lime_analysis=LimeAnalysisResponse(**lime_result) if lime_result else None,
        explainability=legacy_explainability,
        metrics=pipeline_metrics,
    )


# =====================================================================
# EXISTING ENDPOINTS (unchanged)
# =====================================================================

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
    filepath = metrics_tracker.export_results()
    csv_path = metrics_tracker.export_csv()

    return {
        "total_queries": len(metrics_tracker.records),
        "summary": metrics_tracker._compute_summary(),
        "exported_to": filepath,
        "csv_exported_to": csv_path,
        "records": [r.to_dict() for r in metrics_tracker.records[-10:]],
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