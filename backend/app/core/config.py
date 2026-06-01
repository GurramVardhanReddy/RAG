"""
Application settings loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- API Keys ---
    openai_api_key: str = ""

    # --- Vector Database (Qdrant) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "explainable_rag"

    # --- Backend ---
    debug: bool = True
    log_level: str = "info"

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768

    # --- Sparse Embedding (NEW — hybrid search) ---
    sparse_model: str = "prithivida/Splade_PP_en_v1"

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Retrieval ---
    top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Hybrid Search (NEW) ---
    use_hybrid: bool = True
    use_reranker: bool = True
    dense_prefetch: int = 40   # Dense candidates — higher = more weight to dense
    sparse_prefetch: int = 10  # Sparse candidates — lower = less weight to sparse

    # --- File Upload ---
    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf", ".docx", ".txt"]
    upload_dir: str = "data/uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance - loaded once at startup."""
    return Settings()
