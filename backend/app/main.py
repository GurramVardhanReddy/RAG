"""
Explainable RAG — FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.app.core.config import get_settings
from backend.app.api.routes import router as api_router
from backend.app.services.vector_store import VectorStoreService
from backend.rag.embedding.embedder import get_embedding_model

settings = get_settings()


def create_app() -> FastAPI:
    """Application factory."""
    application = FastAPI(
        title="Explainable RAG API",
        description="Information Retrieval Using Explainable Retrieval-Augmented Generation",
        version="0.1.0",
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
        logger.info("Starting Explainable RAG API...")

        # 1. Initialize embedding model (downloads on first run)
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        application.state.embedding_model = get_embedding_model(settings.embedding_model)
        logger.info("Embedding model loaded!")

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

        logger.info("Explainable RAG API ready!")

    # --- Health check ---
    @application.get("/health")
    async def health():
        vs_info = application.state.vector_store.get_collection_info()
        return {
            "status": "healthy",
            "version": "0.1.0",
            "embedding_model": settings.embedding_model,
            "vector_db": vs_info,
        }

    return application


app = create_app()
