"""
Embedding Model Manager — Domain-Adaptive Multi-Model Cache
============================================================
Manages multiple domain-specific embedding models.
Each model is loaded once and cached in memory (singleton per model name).

On document upload:
    1. DomainDetector identifies the domain.
    2. get_embedding_model(model_name) loads the correct domain model.
    3. That model is used for both indexing AND query embedding.

On query:
    1. document_id → domain → model_name is looked up from the registry.
    2. The same model used at index time is used to embed the query.
    This guarantees vector space consistency.

Models and their dimensions:
    nickmuchi/sec-bert-base                           → 768d  (Finance/Regulatory)
    nlpaueb/legal-bert-base-uncased                   → 768d  (Legal)
    dmis-lab/biobert-base-cased-v1.2                  → 768d  (Medical/Biomedical)
    allenai/scibert_scivocab_uncased                  → 768d  (Scientific/Academic)
    microsoft/codebert-base                            → 768d  (Code/Software)
    sentence-transformers/multi-qa-mpnet-base-dot-v1  → 768d  (E-commerce/Product)
    BAAI/bge-m3                                        → 1024d (General fallback)
    BAAI/bge-base-en-v1.5                              → 768d  (Legacy baseline)
"""

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

# ── Multi-model cache ──────────────────────────────────────────────────────────
# Key: model_name (str)  →  Value: HuggingFaceEmbeddings instance
_model_cache: dict[str, HuggingFaceEmbeddings] = {}

# Default / legacy model (used at startup health-check and as fallback)
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """
    Get or initialise a domain-specific embedding model.

    Models are cached globally so each model is downloaded and loaded
    only once per server lifetime, no matter how many documents use it.

    Args:
        model_name: HuggingFace model identifier, e.g. "nickmuchi/sec-bert-base"

    Returns:
        HuggingFaceEmbeddings instance ready for .embed_documents() / .embed_query()
    """
    global _model_cache

    if model_name not in _model_cache:
        logger.info(f"Loading embedding model: {model_name} (first use — downloading if needed)")

        # jinaai/jina-embeddings-v2-base-code uses a custom JinaBERT architecture
        # and requires trust_remote_code=True to load correctly.
        model_kwargs = {"device": "cpu"}
        if "jina" in model_name.lower():
            model_kwargs["trust_remote_code"] = True
            logger.info(f"Enabling trust_remote_code=True for Jina model: {model_name}")

        _model_cache[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},  # Required for cosine similarity
        )
        logger.info(f"Embedding model cached: {model_name}")
    else:
        logger.debug(f"Returning cached embedding model: {model_name}")

    return _model_cache[model_name]


def get_model_for_domain(domain: str) -> HuggingFaceEmbeddings:
    """
    Convenience wrapper — get the embedding model for a detected domain.

    Args:
        domain: One of the 7 domain strings from DomainDetector
                (e.g. "Finance/Regulatory", "Legal", "General")

    Returns:
        Loaded HuggingFaceEmbeddings instance for that domain
    """
    from backend.rag.embedding.domain_detector import DOMAIN_MODEL_MAP
    model_name = DOMAIN_MODEL_MAP.get(domain, "BAAI/bge-m3")
    return get_embedding_model(model_name)


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
) -> list[list[float]]:
    """
    Embed a batch of texts using the specified model.

    Args:
        texts: List of strings to embed
        model_name: HuggingFace model name

    Returns:
        List of float vectors, one per input text
    """
    model = get_embedding_model(model_name)
    return model.embed_documents(texts)


def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL,
) -> list[float]:
    """
    Embed a single query string using the specified model.
    Must use the SAME model that was used to embed the documents,
    otherwise the query vector lives in a different vector space.

    Args:
        query: User's natural language question
        model_name: Must match the model used at index time

    Returns:
        Float vector for the query
    """
    model = get_embedding_model(model_name)
    return model.embed_query(query)


def list_cached_models() -> list[str]:
    """Return names of all currently loaded models."""
    return list(_model_cache.keys())


def get_model_dimension(model_name: str) -> int:
    """
    Return the vector dimension for a given model name.
    Based on known dimensions — no model loading needed.
    """
    from backend.rag.embedding.domain_detector import DOMAIN_DIMENSIONS, DOMAIN_MODEL_MAP
    # Reverse map: model_name → dimension
    model_to_dim = {v: DOMAIN_DIMENSIONS[k] for k, v in DOMAIN_MODEL_MAP.items()}
    return model_to_dim.get(model_name, 768)