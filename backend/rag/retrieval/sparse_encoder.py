"""
Sparse Encoder Module

Generates sparse vector representations for hybrid search using
Qdrant-native sparse vectors. Supports two backends:

    - SPLADE (learned sparse model): Better term importance weighting,
      captures semantic term expansion. Uses "prithivida/Splade_PP_en_v1".
    - BM25 (statistical): Classic term-frequency approach via fastembed.
      Uses "Qdrant/bm25" — no model download needed.

The sparse vectors are stored directly in Qdrant alongside dense vectors,
enabling true single-database hybrid retrieval with RRF fusion.

References:
    - Wang et al., "BERT-based Dense Retrievers Require Interpolation
      with BM25 for Effective Passage Retrieval" (ICTIR 2021)
    - Formal et al., "SPLADE: Sparse Lexical and Expansion Model
      for First Stage Ranking" (SIGIR 2021)
"""

from fastembed import SparseTextEmbedding, SparseEmbedding
from loguru import logger
from typing import Optional

# Module-level cache (singleton pattern, same as dense embedder)
_sparse_model: Optional[SparseTextEmbedding] = None
_current_model_name: Optional[str] = None


def get_sparse_model(
    model_name: str = "prithivida/Splade_PP_en_v1",
) -> SparseTextEmbedding:
    """
    Get or initialize the sparse embedding model (singleton).

    Args:
        model_name: One of:
            - "prithivida/Splade_PP_en_v1" (SPLADE, ~500MB, learned)
            - "Qdrant/bm25" (BM25, statistical, no download)

    Returns:
        SparseTextEmbedding instance
    """
    global _sparse_model, _current_model_name

    if _sparse_model is None or _current_model_name != model_name:
        logger.info(f"Loading sparse embedding model: {model_name}")
        _sparse_model = SparseTextEmbedding(model_name=model_name)
        _current_model_name = model_name
        logger.info(f"Sparse model loaded: {model_name}")

    return _sparse_model


def encode_sparse(
    texts: list[str],
    model_name: str = "prithivida/Splade_PP_en_v1",
) -> list[SparseEmbedding]:
    """
    Generate sparse vectors for a batch of texts.

    Args:
        texts: List of text strings to encode
        model_name: Sparse model to use

    Returns:
        List of SparseEmbedding objects, each with:
            - .indices: numpy array of non-zero dimension indices
            - .values: numpy array of corresponding weights
    """
    model = get_sparse_model(model_name)
    # fastembed returns a generator, convert to list
    embeddings = list(model.embed(texts))
    logger.debug(f"Generated {len(embeddings)} sparse vectors")
    return embeddings


def encode_sparse_query(
    query: str,
    model_name: str = "prithivida/Splade_PP_en_v1",
) -> SparseEmbedding:
    """
    Generate a sparse vector for a single query.

    Args:
        query: Query string
        model_name: Sparse model to use

    Returns:
        SparseEmbedding with .indices and .values
    """
    model = get_sparse_model(model_name)
    embeddings = list(model.query_embed(query))
    return embeddings[0]
