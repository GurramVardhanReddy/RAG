"""
Explainable RAG — FastAPI Application Entry Point

Updated for hybrid retrieval pipeline:
    - Dense encoder (BGE) + Sparse encoder (SPLADE) initialized at startup
    - HybridRetriever instance stored on app.state for use in routes
    - Cross-encoder reranker (BGE-reranker-base) loaded at startup
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.app.core.config import get_settings
from backend.app.api.routes import router as api_router
from backend.app.services.vector_store import VectorStoreService
from backend.rag.embedding.embedder import get_embedding_model
from backend.rag.retrieval.retriever import HybridRetriever

settings = get_settings()


def create_app() -> FastAPI:
    """Application factory."""
    application = FastAPI(
        title="Explainable RAG API",
        description="Information Retrieval Using Explainable Retrieval-Augmented Generation",
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- CORS (allow React frontend to talk to backend) ---
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Register API routes ---
    application.include_router(api_router, prefix="/api/v1")

    # --- Startup event ---
    @application.on_event("startup")
    async def startup():
        logger.info("Starting Explainable RAG API v0.2.0 (Hybrid Retrieval)...")

        # 1. Initialize dense embedding model (downloads on first run)
        logger.info(f"Loading dense embedding model: {settings.embedding_model}")
        application.state.embedding_model = get_embedding_model(settings.embedding_model)
        logger.info("Dense embedding model loaded!")

        # 2. Connect to Qdrant
        vector_store = VectorStoreService(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            embedding_dimension=settings.embedding_dimension,
        )
        vector_store.connect()
        if vector_store.client:
            vector_store.ensure_collection()
        application.state.vector_store = vector_store

        # 3. Initialize the hybrid retriever
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=application.state.embedding_model,
            sparse_model_name=settings.sparse_model,
            reranker_model_name=settings.reranker_model,
            use_hybrid=settings.use_hybrid,
            use_reranker=settings.use_reranker,
            dense_prefetch=settings.dense_prefetch,
            sparse_prefetch=settings.sparse_prefetch,
        )

        # 4. Load sparse encoder (SPLADE)
        if settings.use_hybrid:
            logger.info(f"Loading sparse model: {settings.sparse_model}")
            retriever.init_sparse_encoder()
            logger.info("Sparse encoder (SPLADE) loaded!")

        # 5. Load cross-encoder reranker
        if settings.use_reranker:
            logger.info(f"Loading reranker: {settings.reranker_model}")
            retriever.init_reranker()
            logger.info("Cross-encoder reranker loaded!")

        application.state.retriever = retriever

        logger.info("Explainable RAG API ready! (Hybrid: Dense + Sparse → RRF → Rerank)")

    # --- Health check ---
    @application.get("/health")
    async def health():
        vs_info = application.state.vector_store.get_collection_info()
        return {
            "status": "healthy",
            "version": "0.2.0",
            "embedding_model": settings.embedding_model,
            "sparse_model": settings.sparse_model if settings.use_hybrid else "disabled",
            "reranker_model": settings.reranker_model if settings.use_reranker else "disabled",
            "retrieval_mode": "hybrid_rrf_reranked" if (settings.use_hybrid and settings.use_reranker)
                             else "hybrid_rrf" if settings.use_hybrid
                             else "dense_only",
            "vector_db": vs_info,
        }

    return application


app = create_app()
