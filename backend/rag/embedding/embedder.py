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

BGE-M3 (General domain fallback) additionally supports sparse embeddings
via BGEM3Embeddings wrapper. Domain-specific models are dense-only.

Models and their dimensions:
    FinLang/finance-embeddings-investopedia            → 768d  (Finance/Regulatory)
    Stern5497/sbert-legal-xlm-roberta-base             → 768d  (Legal)
    NeuML/pubmedbert-base-embeddings                   → 768d  (Medical/Biomedical)
    allenai/scibert_scivocab_uncased                   → 768d  (Scientific/Academic)
    jinaai/jina-embeddings-v2-base-code                → 768d  (Code/Software)
    sentence-transformers/multi-qa-mpnet-base-dot-v1   → 768d  (E-commerce/Product)
    BAAI/bge-m3                                         → 1024d (General fallback)
"""

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from typing import Optional

# ── Multi-model cache ──────────────────────────────────────────────────────────
# Key: model_name (str)  →  Value: model instance (HuggingFaceEmbeddings or BGEM3Embeddings)
_model_cache: dict = {}

DEFAULT_MODEL = "BAAI/bge-m3"


# ── BGE-M3 wrapper (dense + sparse) ──────────────────────────────────────────

class BGEM3Embeddings:
    """
    Wrapper around FlagEmbedding's BGEM3FlagModel exposing:
      - embed_documents(texts) / embed_query(text)  — LangChain-compatible dense
      - embed_documents_dense(texts)                — explicit dense
      - embed_documents_sparse(texts)               — BGE-M3 lexical weights
      - embed_query_sparse(text)                    — sparse for a single query
    """

    def __init__(self, model_name_or_path: str = "BAAI/bge-m3", use_fp16: bool = False):
        from FlagEmbedding import BGEM3FlagModel
        logger.info(f"Loading BGE-M3: {model_name_or_path} (fp16={use_fp16})")
        self.model = BGEM3FlagModel(model_name_or_path=model_name_or_path, use_fp16=use_fp16)
        self.model_name = model_name_or_path
        logger.info("BGE-M3 loaded successfully")

    def embed_documents_dense(self, texts: list[str]) -> list[list[float]]:
        output = self.model.encode(texts, batch_size=12, max_length=8192, return_dense=True)
        return output["dense_vecs"].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents_dense(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents_dense([text])[0]

    def embed_documents_sparse(self, texts: list[str]) -> list[dict]:
        output = self.model.encode(texts, batch_size=12, max_length=8192, return_sparse=True)
        result = []
        for sv in output["lexical_weights"]:
            result.append({
                "indices": [int(k) for k in sv.keys()],
                "values":  [float(v) for v in sv.values()],
            })
        return result

    def embed_query_sparse(self, text: str) -> dict:
        return self.embed_documents_sparse([text])[0]


# ── Model loader ───────────────────────────────────────────────────────────────

def get_embedding_model(model_name: str = DEFAULT_MODEL):
    """
    Get or initialise a domain-specific embedding model (cached after first load).

    BGE-M3 → BGEM3Embeddings (dense + sparse)
    All others → HuggingFaceEmbeddings (dense only)
    """
    global _model_cache

    if model_name not in _model_cache:
        logger.info(f"Loading embedding model: {model_name}")

        if "bge-m3" in model_name.lower():
            _model_cache[model_name] = BGEM3Embeddings(model_name_or_path=model_name, use_fp16=False)
        else:
            model_kwargs = {"device": "cpu"}
            if "jina" in model_name.lower():
                model_kwargs["trust_remote_code"] = True
            _model_cache[model_name] = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True},
            )
        logger.info(f"Embedding model cached: {model_name}")

    return _model_cache[model_name]


def get_model_for_domain(domain: str):
    """Get the embedding model for a detected domain."""
    from backend.rag.embedding.domain_detector import DOMAIN_MODEL_MAP
    model_name = DOMAIN_MODEL_MAP.get(domain, DEFAULT_MODEL)
    return get_embedding_model(model_name)


def list_cached_models() -> list[str]:
    """Return names of all currently loaded models."""
    return list(_model_cache.keys())


def get_model_dimension(model_name: str) -> int:
    """Return the vector dimension for a given model name."""
    from backend.rag.embedding.domain_detector import DOMAIN_DIMENSIONS, DOMAIN_MODEL_MAP
    model_to_dim = {v: DOMAIN_DIMENSIONS[k] for k, v in DOMAIN_MODEL_MAP.items()}
    return model_to_dim.get(model_name, 768)


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    return get_embedding_model(model_name).embed_documents(texts)


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> list[float]:
    return get_embedding_model(model_name).embed_query(query)
