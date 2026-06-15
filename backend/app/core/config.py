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

    # --- Embedding (BGE-M3: unified dense + sparse) ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Retrieval ---
    top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_threshold: float = 0.1
    # NOTE: embedding_dimension changed 768→1024 with BGE-M3.
    # Delete the existing Qdrant collection before first run.

    # --- Hybrid Search ---
    use_hybrid: bool = True
    use_reranker: bool = True
    dense_prefetch: int = 40
    sparse_prefetch: int = 10

    # --- HyDE ---
    use_hyde: bool = False

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
