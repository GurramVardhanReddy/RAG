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
            reranker_model_name=settings.reranker_model,
            reranker_threshold=settings.reranker_threshold,
            use_hybrid=settings.use_hybrid,
            use_reranker=settings.use_reranker,
            use_hyde=settings.use_hyde,
            dense_prefetch=settings.dense_prefetch,
            sparse_prefetch=settings.sparse_prefetch,
            openai_api_key=settings.openai_api_key,
        )

        # 4. Load cross-encoder reranker
        if settings.use_reranker:
            logger.info(f"Loading reranker: {settings.reranker_model}")
            retriever.init_reranker()
            logger.info("Reranker loaded!")

        application.state.retriever = retriever

        reranker_label = f"{settings.reranker_type}:{settings.reranker_model}" if settings.use_reranker else "off"
        logger.info(f"Explainable RAG API ready! (BGE-M3 dense+sparse → RRF → reranker:{reranker_label})")

    # --- Health check ---
    @application.get("/health")
    async def health():
        vs_info = application.state.vector_store.get_collection_info()
        return {
            "status": "healthy",
            "version": "0.2.0",
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model if settings.use_reranker else "disabled",
            "retrieval_mode": (
                "hyde_hybrid_rrf" if (settings.use_hyde and settings.use_hybrid) else
                "hyde_dense" if settings.use_hyde else
                "hybrid_rrf_reranked" if (settings.use_hybrid and settings.use_reranker) else
                "hybrid_rrf" if settings.use_hybrid else
                "dense_only"
            ),
            "vector_db": vs_info,
        }

    return application


app = create_app()
