"""
Explainable RAG — FastAPI Application Entry Point (v0.3.0)
==========================================================
Domain-Adaptive Embedding Selection update:

    - No single global embedding model at startup
    - Domain models are loaded on-demand per document (cached after first use)
    - VectorStoreService now manages per-domain collections
    - HybridRetriever receives shared document_registry for domain lookup
    - SPLADE sparse encoder + BGE cross-encoder reranker still loaded at startup

Startup sequence:
    1. Connect to Qdrant
    2. Load SPLADE sparse encoder
    3. Load BGE cross-encoder reranker
    4. HybridRetriever initialised (no embedding model — loaded per-document)

Domain models are downloaded from HuggingFace on FIRST USE of each domain.
Subsequent uploads/queries for the same domain use the cached model.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.app.core.config import get_settings
from backend.app.api.routes import router as api_router, document_registry
from backend.app.services.vector_store import VectorStoreService
from backend.rag.retrieval.retriever import HybridRetriever

settings = get_settings()


def create_app() -> FastAPI:
    """Application factory."""
    application = FastAPI(
        title="Explainable RAG API",
        description=(
            "Information Retrieval Using Explainable Retrieval-Augmented Generation. "
            "v0.3.0: Domain-Adaptive Embedding Selection with XAI attribution."
        ),
        version="0.3.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    application.include_router(api_router, prefix="/api/v1")

    # ── Startup ───────────────────────────────────────────────────────────────
    @application.on_event("startup")
    async def startup():
        logger.info("=" * 60)
        logger.info("Starting Explainable RAG API v0.3.0")
        logger.info("Domain-Adaptive Embedding Selection enabled")
        logger.info("=" * 60)

        # 1. Connect to Qdrant
        vector_store = VectorStoreService(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_prefix=settings.qdrant_collection_name,
        )
        vector_store.connect()
        application.state.vector_store = vector_store
        logger.info("Qdrant connected. Domain collections created on first upload.")

        # 2. Initialise hybrid retriever (no global embedding model)
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=None,           # Domain models loaded per-document
            sparse_model_name=settings.sparse_model,
            reranker_model_name=settings.reranker_model,
            use_hybrid=settings.use_hybrid,
            use_reranker=settings.use_reranker,
            prefetch_limit=settings.prefetch_limit,
            document_registry=document_registry,  # Shared live registry
        )

        # 3. Load SPLADE sparse encoder
        if settings.use_hybrid:
            logger.info(f"Loading sparse encoder: {settings.sparse_model}")
            retriever.init_sparse_encoder()
            logger.info("SPLADE sparse encoder ready.")

        # 4. Load cross-encoder reranker
        if settings.use_reranker:
            logger.info(f"Loading cross-encoder reranker: {settings.reranker_model}")
            retriever.init_reranker()
            logger.info("Cross-encoder reranker ready.")

        application.state.retriever = retriever

        logger.info("=" * 60)
        logger.info("Explainable RAG API v0.3.0 ready!")
        logger.info("Domain models loaded on first upload (per domain).")
        logger.info(
            f"Retrieval: {'Hybrid RRF + Rerank' if settings.use_hybrid and settings.use_reranker else 'Dense-only'}"
        )
        logger.info("=" * 60)

    # ── Health check ──────────────────────────────────────────────────────────
    @application.get("/health")
    async def health():
        from backend.rag.embedding.embedder import list_cached_models
        from backend.rag.embedding.domain_detector import DOMAIN_MODEL_MAP

        return {
            "status": "healthy",
            "version": "0.3.0",
            "feature": "domain_adaptive_embedding",
            "domain_model_map": DOMAIN_MODEL_MAP,
            "cached_models": list_cached_models(),
            "retrieval_mode": (
                "hybrid_rrf_reranked" if (settings.use_hybrid and settings.use_reranker)
                else "hybrid_rrf" if settings.use_hybrid
                else "dense_only"
            ),
            "vector_db": {
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "collection_prefix": settings.qdrant_collection_name,
            },
        }

    return application


app = create_app()
