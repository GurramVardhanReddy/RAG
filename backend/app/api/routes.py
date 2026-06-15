"""
API Routes — Domain-Adaptive Explainable RAG
=============================================
Updated with:
    1. Domain detection at upload time (DomainDetector)
    2. Domain-specific embedding model selection
    3. XAI explanation of domain decision in UploadResponse
    4. Domain-aware query routing (same model for index + query)
    5. Per-domain Qdrant collections

Every upload now:
    → Detects domain (Finance/Legal/Medical/Scientific/Code/Ecommerce/General)
    → Selects correct embedding model for that domain
    → Stores chunks in the domain-specific Qdrant collection
    → Returns XAI explanation of WHY that domain + model was chosen

Every query now:
    → Looks up document domain from registry
    → Embeds query with SAME model used at index time
    → Searches in the correct domain collection
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
from backend.rag.generation.generator import generate_answer
from backend.rag.explainability.engine import ExplainabilityEngine
from backend.rag.retrieval.sparse_encoder import encode_sparse
from backend.rag.metrics.tracker import RAGMetricsTracker, ResponseQualityMetrics, ResourceMetrics
from backend.rag.multimodal.processor import extract_and_embed_visuals
from backend.rag.embedding.domain_detector import detect_domain, DomainDetectionResult
from backend.rag.embedding.embedder import get_embedding_model

router = APIRouter()
settings = get_settings()

# In-memory document registry
# Stores: filename, pages, chunks, domain, embedding_model, xai info, etc.
document_registry: dict[str, dict] = {}

# Global metrics tracker
metrics_tracker = RAGMetricsTracker(output_dir="./metrics_output")

# Explainability engine (three-layer)
explainability_engine = ExplainabilityEngine()


# =====================================================================
# REQUEST / RESPONSE MODELS
# =====================================================================

class QueryRequest(BaseModel):
    question: str
    document_id: str
    top_k: int = 5
    run_perturbation_test: bool = False
    run_lime_analysis: bool = False


class SourceChunk(BaseModel):
    content: str
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    paragraph_numbers: list[int] = []
    paragraph_match_type: str = "unknown"
    chunk_index: int = 0
    relevance_score: float = 0.0
    retrieval_method: str = "hybrid_rrf_reranked"


class SourceAttributionResponse(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    cited_chunk_count: int = 0
    relevant_chunk_count: int = 0
    correctly_cited: int = 0


class ProvenanceCompletenessResponse(BaseModel):
    completeness_score: float = 0.0
    pages_covered: list[int] = []
    paragraphs_covered: list[int] = []
    unique_pages: int = 0
    unique_paragraphs: int = 0
    has_page_level: bool = False
    has_paragraph_level: bool = False


class ConfidenceCalibrationResponse(BaseModel):
    confidence_score: float = 0.0
    confidence_tier: str = "low"
    calibration_error: float = 0.0
    score_variance: float = 0.0
    mean_retrieval_score: float = 0.0
    max_retrieval_score: float = 0.0


class ExplanationFaithfulnessResponse(BaseModel):
    overall_faithfulness: float = 0.0
    answer_grounding_ratio: float = 0.0
    grounded_tokens: int = 0
    total_answer_tokens: int = 0
    source_coverage: float = 0.0
    # Perturbation test fields — populated when run_perturbation_test=True
    perturbation_tested: bool = False
    chunk_faithfulness: list[dict] = []


class QuantitativeMetrics(BaseModel):
    source_attribution: SourceAttributionResponse
    provenance_completeness: ProvenanceCompletenessResponse
    confidence_calibration: ConfidenceCalibrationResponse
    explanation_faithfulness: ExplanationFaithfulnessResponse


class ChunkRelevanceHeatmapResponse(BaseModel):
    query: str = ""
    chunks: list[dict] = []
    scores: list[float] = []
    score_distribution: dict = {}
    max_score: float = 0.0
    min_score: float = 0.0
    mean_score: float = 0.0
    score_spread: float = 0.0


class ProvenanceTraceResponse(BaseModel):
    query: str = ""
    retrieval_method: str = ""
    retrieval_stages: list[dict] = []
    chunk_to_answer_mapping: list[dict] = []
    answer_with_citations: str = ""
    provenance_chain: list[dict] = []


class VisualizationData(BaseModel):
    chunk_relevance_heatmap: ChunkRelevanceHeatmapResponse
    provenance_trace: ProvenanceTraceResponse


class ExpertEvaluationResponse(BaseModel):
    query: str = ""
    answer: str = ""
    source_citations: list[dict] = []
    evaluation_dimensions: dict = {}


class LimeAnalysisResponse(BaseModel):
    word_importance: list[dict] = []
    query_sensitivity: float = 0.0
    most_important_words: list[dict] = []
    least_important_words: list[dict] = []
    stability_score: float = 0.0
    total_perturbations: int = 0
    original_query: str = ""


class ThreeLayerExplainabilityResponse(BaseModel):
    summary: dict = {}
    layer1_quantitative: QuantitativeMetrics
    layer2_visualization: VisualizationData
    layer3_expert_evaluation: ExpertEvaluationResponse


class ExplainabilityInfo(BaseModel):
    sources_used: list[SourceChunk]
    confidence_score: float
    retrieval_strategy: str
    total_chunks_searched: int
    processing_time_ms: float
    retrieval_timings: Optional[dict] = None


class MetricsInfo(BaseModel):
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
    answer: str
    three_layer_explainability: ThreeLayerExplainabilityResponse
    lime_analysis: Optional[LimeAnalysisResponse] = None
    explainability: ExplainabilityInfo
    metrics: Optional[MetricsInfo] = None


# ── NEW: Domain XAI in UploadResponse ────────────────────────────────────────

class DomainXAIResponse(BaseModel):
    """XAI explanation for domain detection — returned on every upload."""
    detected_domain: str
    embedding_model: str
    embedding_dimension: int
    confidence: float
    domain_probabilities: dict          # All 7 domain scores (softmax)
    top_keywords: list[dict]            # [{word, importance_score, frequency, context_snippet}]
    xai_explanation: str                # Human-readable explanation
    xai_method: str
    gpt_used: bool
    gpt_reasoning: str


class UploadResponse(BaseModel):
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
    # Domain-adaptive fields (NEW)
    domain_detection: DomainXAIResponse


# =====================================================================
# ENDPOINTS
# =====================================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF/DOCX document.

    NEW: Automatically detects the document domain and selects the
    appropriate domain-specific embedding model. Returns XAI attribution
    explaining WHY the domain and model were chosen.

    Pipeline:
        File → Text extraction → Domain detection (XAI)
             → Domain embedding model selection
             → Chunk → Dense embed (domain model) + Sparse embed (SPLADE)
             → Store in domain-specific Qdrant collection
    """
    start_time = time.time()

    # 1. Validate file type
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Use .pdf, .docx, or .txt"
        )

    # 2. Save to disk
    document_id = str(uuid.uuid4())[:8]
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, f"{document_id}_{filename}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    logger.info(f"Saved '{filename}' as '{save_path}'")

    # 3. Load document (with paragraph detection)
    docs = load_document(save_path)

    # Build page_paragraphs map
    page_paragraphs = {}
    total_paragraphs = 0
    for doc in docs:
        page_num = doc.metadata.get("page", 0)
        paragraphs = doc.metadata.get("paragraphs", [])
        page_paragraphs[page_num] = paragraphs
        total_paragraphs += len(paragraphs)

    # 4. ── DOMAIN DETECTION (NEW) ─────────────────────────────────────────
    full_text = "\n".join(doc.page_content for doc in docs)
    logger.info(f"Running domain detection on {len(full_text)} chars of text...")

    domain_result: DomainDetectionResult = detect_domain(
        text=full_text,
        openai_api_key=settings.openai_api_key,
    )

    detected_domain = domain_result.detected_domain
    model_name = domain_result.embedding_model
    embedding_dim = domain_result.embedding_dimension

    logger.info(
        f"Domain: '{detected_domain}' | Model: '{model_name}' | "
        f"Confidence: {domain_result.confidence:.3f}"
    )

    # 5. Load domain-specific embedding model
    embedding_model = get_embedding_model(model_name)

    # 6. Chunk the document
    chunks = semantic_chunk(docs, embedding_model=embedding_model)

    # 7. Generate DENSE embeddings (domain-specific model)
    chunk_texts = [chunk.page_content for chunk in chunks]
    dense_vectors = embedding_model.embed_documents(chunk_texts)
    logger.info(f"Generated {len(dense_vectors)} dense vectors (model={model_name})")

    # 8. Generate SPARSE embeddings (SPLADE — model-agnostic)
    sparse_vectors = encode_sparse(chunk_texts, model_name=settings.sparse_model)
    logger.info(f"Generated {len(sparse_vectors)} sparse vectors")

    # 9. Store in domain-specific Qdrant collection
    vector_store: VectorStoreService = request.app.state.vector_store
    vector_store.add_documents(
        chunks=chunks,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        document_id=document_id,
        page_paragraphs=page_paragraphs,
        domain=detected_domain,
        embedding_model_name=model_name,
    )

    # 10. Multimodal extraction (PDFs only)
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
                domain=detected_domain,
                embedding_model_name=model_name,
            )
        except Exception as e:
            logger.warning(f"Multimodal extraction failed (non-fatal): {e}")

    # 11. Register document with domain metadata
    processing_time = (time.time() - start_time) * 1000
    document_registry[document_id] = {
        "filename": filename,
        "total_pages": len(docs),
        "total_chunks": len(chunks),
        "total_paragraphs": total_paragraphs,
        "processing_time_ms": processing_time,
        # Domain-adaptive metadata (NEW)
        "domain": detected_domain,
        "embedding_model": model_name,
        "embedding_dimension": embedding_dim,
        "domain_confidence": domain_result.confidence,
        "vector_types": [f"dense ({model_name})", "sparse (SPLADE)"],
        "visual_elements": visual_stats,
    }

    # 12. Update retriever's document registry reference
    request.app.state.retriever.document_registry = document_registry

    # 13. Build XAI response
    top_keywords_dicts = [
        {
            "word": m.word,
            "importance_score": m.importance_score,
            "frequency": m.frequency,
            "context_snippet": m.context_snippet,
        }
        for m in domain_result.top_keywords
    ]

    domain_xai = DomainXAIResponse(
        detected_domain=detected_domain,
        embedding_model=model_name,
        embedding_dimension=embedding_dim,
        confidence=domain_result.confidence,
        domain_probabilities=domain_result.domain_probabilities,
        top_keywords=top_keywords_dicts,
        xai_explanation=domain_result.xai_explanation,
        xai_method=domain_result.xai_method,
        gpt_used=domain_result.gpt_used,
        gpt_reasoning=domain_result.gpt_reasoning,
    )

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        total_pages=len(docs),
        total_chunks=len(chunks) + visual_stats["chunks_added"],
        total_paragraphs=total_paragraphs,
        processing_time_ms=round(processing_time, 2),
        message=(
            f"Document processed: domain='{detected_domain}', "
            f"model='{model_name}', "
            f"{len(docs)} pages, {len(chunks)} text chunks, "
            f"{visual_stats['tables']} tables, {visual_stats['images']} images"
        ),
        visual_tables=visual_stats["tables"],
        visual_images=visual_stats["images"],
        visual_chunks=visual_stats["chunks_added"],
        domain_detection=domain_xai,
    )


@router.post("/query", response_model=QueryResponse)
async def query_document(request: Request, body: QueryRequest):
    """
    Ask a question about an uploaded document.

    NEW: Query is embedded using the SAME domain-specific model
    that was used to index the document, ensuring vector space consistency.

    Pipeline:
        Question → look up domain for document_id
                → embed with domain model
                → hybrid search in domain collection
                → RRF → Rerank → GPT-4o → Answer + Explainability
    """
    metrics_record = metrics_tracker.start_query(body.question)

    vector_store: VectorStoreService = request.app.state.vector_store
    retriever = request.app.state.retriever

    # ── STAGE 1: HYBRID RETRIEVAL ─────────────────────────────────────────────
    with metrics_tracker.timer.time_stage("query_to_vector"):
        pass  # Done inside retriever.retrieve()

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

    # ── STAGE 2: GENERATION ───────────────────────────────────────────────────
    with metrics_tracker.timer.time_stage("generation"):
        # Select prompt template based on detected domain
        # Finance/Regulatory → "domain" (UCITS compliance expert prompt)
        # Scientific/Academic → "general" (concise factual QA)
        # Legal → "domain" (compliance-style, source-heavy)
        # Medical/Biomedical → "general" (factual, source-cited)
        # Code/Software → "general" (concise technical QA)
        # E-commerce/Product → "general" (concise product QA)
        # General → "general" (default QA)
        doc_domain = document_registry.get(body.document_id, {}).get("domain", "General")
        DOMAIN_PROMPT_MODE = {
            "Finance/Regulatory":  "domain",    # UCITS compliance expert persona
            "Legal":               "domain",    # Compliance-style, source-heavy
            "Medical/Biomedical":  "general",   # Factual, concise
            "Scientific/Academic": "general",   # Factual, concise
            "Code/Software":       "general",   # Technical QA
            "E-commerce/Product":  "general",   # Product QA
            "General":             "general",   # Default
        }
        prompt_mode = DOMAIN_PROMPT_MODE.get(doc_domain, "general")
        logger.info(f"Generation prompt mode: '{prompt_mode}' (domain='{doc_domain}')")

        answer = generate_answer(
            question=body.question,
            search_results=search_results,
            model_name="gpt-4o",
            temperature=0,
            api_key=settings.openai_api_key,
            prompt_mode=prompt_mode,
        )

    # ── STAGE 3: EXPLAINABILITY ───────────────────────────────────────────────
    with metrics_tracker.timer.time_stage("explainability"):
        doc_info = document_registry.get(body.document_id, {})
        total_chunks = doc_info.get("total_chunks", 0)

        xai_result = explainability_engine.analyze(
            question=body.question,
            answer=answer,
            search_results=search_results,
            retrieval_metadata=retrieval_metadata,
            total_chunks=total_chunks,
        )

    # ── STAGE 4: PERTURBATION TEST (optional, expensive) ─────────────────────
    # Removes each chunk one-by-one, re-generates, measures answer change.
    # Reference: Counterfactual provenance testing (Rorseth et al., RAGE 2024)
    if body.run_perturbation_test:
        try:
            logger.info("Running perturbation-based faithfulness test...")

            def _generate_fn(q: str, results: list) -> str:
                return generate_answer(
                    question=q,
                    search_results=results,
                    model_name="gpt-4o",
                    temperature=0,
                    api_key=settings.openai_api_key,
                    prompt_mode=prompt_mode,
                )

            perturbation_results = explainability_engine.run_perturbation_test(
                question=body.question,
                search_results=search_results,
                original_answer=answer,
                generate_fn=_generate_fn,
            )

            if perturbation_results:
                faithful_count = sum(1 for r in perturbation_results if r.get("is_faithful"))
                xai_result.explanation_faithfulness.perturbation_tested = True
                xai_result.explanation_faithfulness.overall_faithfulness = (
                    faithful_count / len(perturbation_results)
                )
                xai_result.explanation_faithfulness.chunk_faithfulness = perturbation_results
                logger.info(
                    f"Perturbation test complete — "
                    f"{faithful_count}/{len(perturbation_results)} chunks faithful"
                )
        except Exception as e:
            logger.warning(f"Perturbation test failed (non-fatal): {e}")

    # ── STAGE 4: LIME (optional) ──────────────────────────────────────────────
    # LIME applied to retrieval: which query words drove chunk selection?
    # Reference: Ribeiro et al. (2016) — LIME, model-agnostic local explanations
    lime_response = None
    if body.run_lime_analysis:
        try:
            logger.info("Running LIME analysis on retrieval query...")

            def _retrieve_fn(q: str, k: int) -> list:
                """Lightweight retrieval wrapper for LIME perturbations."""
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
            # Add original_query field if not present
            lime_result["original_query"] = body.question
            lime_result["total_perturbations"] = len(lime_result.get("word_importance", []))
            lime_response = LimeAnalysisResponse(**lime_result)
            logger.info(f"LIME complete — {lime_result['total_perturbations']} perturbations, "
                        f"sensitivity={lime_result.get('query_sensitivity', 0):.3f}")
        except Exception as e:
            logger.warning(f"LIME analysis failed (non-fatal): {e}")

    # ── BUILD RESPONSE ────────────────────────────────────────────────────────
    xai_dict = xai_result.to_dict()

    ef = xai_dict["explanation_faithfulness"]
    layer1 = QuantitativeMetrics(
        source_attribution=SourceAttributionResponse(**xai_dict["source_attribution"]),
        provenance_completeness=ProvenanceCompletenessResponse(**xai_dict["provenance_completeness"]),
        confidence_calibration=ConfidenceCalibrationResponse(**xai_dict["confidence_calibration"]),
        explanation_faithfulness=ExplanationFaithfulnessResponse(
            overall_faithfulness=ef.get("overall_faithfulness", 0.0),
            answer_grounding_ratio=ef.get("answer_grounding_ratio", 0.0),
            grounded_tokens=ef.get("grounded_tokens", 0),
            total_answer_tokens=ef.get("total_answer_tokens", 0),
            source_coverage=ef.get("source_coverage", 0.0),
            perturbation_tested=ef.get("perturbation_tested", False),
            chunk_faithfulness=ef.get("chunk_faithfulness", []),
        ),
    )

    layer2 = VisualizationData(
        chunk_relevance_heatmap=ChunkRelevanceHeatmapResponse(**xai_dict["chunk_relevance_heatmap"]),
        provenance_trace=ProvenanceTraceResponse(**xai_dict["provenance_trace"]),
    )

    layer3 = ExpertEvaluationResponse(**xai_dict["expert_evaluation"])

    three_layer = ThreeLayerExplainabilityResponse(
        summary=xai_result.summary(),
        layer1_quantitative=layer1,
        layer2_visualization=layer2,
        layer3_expert_evaluation=layer3,
    )

    sources = [
        SourceChunk(
            content=r["text"],
            page_number=r.get("page_number"),
            paragraph_number=r.get("paragraph_number"),
            paragraph_numbers=r.get("paragraph_numbers", []),
            paragraph_match_type=r.get("paragraph_match_type", "unknown"),
            chunk_index=r.get("chunk_index", 0),
            relevance_score=r.get("score", 0.0),
            retrieval_method=r.get("retrieval_method", retrieval_metadata.get("retrieval_method", "")),
        )
        for r in search_results
    ]

    timings = metrics_tracker.timer.get_all_timings()
    end_to_end_ms = sum(timings.values())

    return QueryResponse(
        answer=answer,
        three_layer_explainability=three_layer,
        lime_analysis=lime_response,
        explainability=ExplainabilityInfo(
            sources_used=sources,
            confidence_score=xai_dict["confidence_calibration"]["confidence_score"],
            retrieval_strategy=retrieval_metadata.get("retrieval_method", "hybrid"),
            total_chunks_searched=total_chunks,
            processing_time_ms=round(end_to_end_ms, 2),
            retrieval_timings={
                **retrieval_metadata.get("timings", {}),
                "domain": retrieval_metadata.get("domain", "General"),
                "query_embedding_model": retrieval_metadata.get("query_embedding_model", "unknown"),
            },
        ),
        metrics=MetricsInfo(
            query_to_vector_ms=round(timings.get("query_to_vector", 0), 2),
            similarity_search_ms=round(timings.get("similarity_search", 0), 2),
            generation_ms=round(timings.get("generation", 0), 2),
            explainability_ms=round(timings.get("explainability", 0), 2),
            end_to_end_ms=round(end_to_end_ms, 2),
        ),
    )


@router.get("/documents")
async def list_documents():
    """List all uploaded documents with their domain metadata."""
    return {
        "documents": [
            {
                "document_id": doc_id,
                "filename": info.get("filename", ""),
                "total_pages": info.get("total_pages", 0),
                "total_chunks": info.get("total_chunks", 0),
                "domain": info.get("domain", "General"),
                "embedding_model": info.get("embedding_model", "unknown"),
                "domain_confidence": info.get("domain_confidence", 0.0),
                "processing_time_ms": info.get("processing_time_ms", 0.0),
            }
            for doc_id, info in document_registry.items()
        ],
        "total": len(document_registry),
    }


@router.get("/health/collections")
async def collection_health(request: Request):
    """Check status of all domain-specific Qdrant collections."""
    vector_store: VectorStoreService = request.app.state.vector_store
    return {"collections": vector_store.get_all_collection_info()}