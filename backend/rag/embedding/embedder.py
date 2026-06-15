"""
Embedding Model Manager — BGE-M3

BGE-M3 unifies dense, sparse (lexical), and ColBERT (multi-vector)
representations in a single model, replacing the previous two-model
setup (BGE-base for dense + SPLADE for sparse).

embed_documents() / embed_query() are kept as LangChain-compatible
aliases to the dense path so SemanticChunker and other LangChain
components continue to work without modification.
"""

from loguru import logger
from typing import Optional

_model: Optional[object] = None


class BGEM3Embeddings:
    """
    Thin wrapper around FlagEmbedding's FlagModel that exposes:

      Dense (LangChain-compatible):
        embed_documents(texts)   -> list[list[float]]
        embed_query(text)        -> list[float]

      Dense (explicit):
        embed_documents_dense(texts) -> list[list[float]]

      Sparse (BGE-M3 lexical weights):
        embed_documents_sparse(texts) -> list[{"indices": [...], "values": [...]}]
        embed_query_sparse(text)      -> {"indices": [...], "values": [...]}
    """

    def __init__(self, model_name_or_path: str = "BAAI/bge-m3", use_fp16: bool = False):
        from FlagEmbedding import BGEM3FlagModel
        logger.info(f"Loading BGE-M3: {model_name_or_path} (fp16={use_fp16})")
        self.model = BGEM3FlagModel(model_name_or_path=model_name_or_path, use_fp16=use_fp16)
        self.model_name = model_name_or_path
        logger.info("BGE-M3 loaded successfully")

    # ------------------------------------------------------------------
    # Dense
    # ------------------------------------------------------------------

    def embed_documents_dense(self, texts: list[str]) -> list[list[float]]:
        output = self.model.encode(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
        )
        return output["dense_vecs"].tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents_dense([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents_dense(texts)

    # ------------------------------------------------------------------
    # Sparse (lexical weights)
    # ------------------------------------------------------------------

    def embed_documents_sparse(self, texts: list[str]) -> list[dict]:
        """
        Returns a list of dicts {"indices": [int, ...], "values": [float, ...]}.
        BGE-M3 lexical_weights maps token-id (int) -> weight (float).
        """
        output = self.model.encode(
            texts,
            batch_size=12,
            max_length=8192,
            return_sparse=True,
        )
        sparse_vecs = output["lexical_weights"]
        result = []
        for sv in sparse_vecs:
            indices = [int(k) for k in sv.keys()]
            values  = [float(v) for v in sv.values()]
            result.append({"indices": indices, "values": values})
        return result

    def embed_query_sparse(self, text: str) -> dict:
        return self.embed_documents_sparse([text])[0]


def get_embedding_model(model_name: str = "BAAI/bge-m3") -> BGEM3Embeddings:
    """
    Return the singleton BGE-M3 model, initialising it on first call.
    """
    global _model
    if _model is None:
        _model = BGEM3Embeddings(model_name_or_path=model_name, use_fp16=False)
    return _model


def embed_texts(texts: list[str], model_name: str = "BAAI/bge-m3") -> list[list[float]]:
    return get_embedding_model(model_name).embed_documents_dense(texts)


def embed_query(query: str, model_name: str = "BAAI/bge-m3") -> list[float]:
    return get_embedding_model(model_name).embed_query(query)
