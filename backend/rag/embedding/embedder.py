"""
Embedding Model Manager

Handles initialization and caching of embedding models.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

# Module-level cache to avoid reloading the model
_embedding_model = None


def get_embedding_model(model_name: str = "BAAI/bge-base-en-v1.5") -> HuggingFaceEmbeddings:
    """
    Get or initialize the embedding model (singleton pattern).
    The model is downloaded once and cached in memory.
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # Change to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},  # Important for cosine similarity
        )
        logger.info("Embedding model loaded successfully")

    return _embedding_model


def embed_texts(texts: list[str], model_name: str = "BAAI/bge-base-en-v1.5") -> list[list[float]]:
    """Embed a batch of texts and return their vectors."""
    model = get_embedding_model(model_name)
    return model.embed_documents(texts)


def embed_query(query: str, model_name: str = "BAAI/bge-base-en-v1.5") -> list[float]:
    """Embed a single query string."""
    model = get_embedding_model(model_name)
    return model.embed_query(query)
