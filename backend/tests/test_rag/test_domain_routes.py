"""Domain → embedding model routes, and query-time model lookup for older documents."""

from backend.rag.embedding.domain_detector import DOMAIN_DIMENSIONS, DOMAIN_MODEL_MAP
from backend.rag.retrieval.retriever import GENERAL_MODEL, HybridRetriever


def test_medical_and_scientific_use_general_model():
    # BEIR showed the General model beating PubMedBERT (NFCorpus) and SciBERT (SciFact)
    assert DOMAIN_MODEL_MAP["Medical/Biomedical"] == DOMAIN_MODEL_MAP["General"]
    assert DOMAIN_MODEL_MAP["Scientific/Academic"] == DOMAIN_MODEL_MAP["General"]


def test_general_model_matches_retriever_default():
    assert DOMAIN_MODEL_MAP["General"] == GENERAL_MODEL


def test_every_domain_has_a_dimension():
    assert set(DOMAIN_MODEL_MAP) == set(DOMAIN_DIMENSIONS)


def test_document_keeps_the_model_it_was_indexed_with():
    # A document indexed with the former Scientific route must still be queried with it
    registry = {
        "old-doc": {
            "domain": "Scientific/Academic",
            "embedding_model": "allenai/scibert_scivocab_uncased",
        }
    }
    retriever = HybridRetriever(vector_store=None, document_registry=registry)
    assert retriever._get_domain_info("old-doc") == (
        "Scientific/Academic",
        "allenai/scibert_scivocab_uncased",
    )
    assert retriever._get_domain_info("unknown") == ("General", GENERAL_MODEL)
