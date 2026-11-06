"""
Tests for the BM25 sparse encoder.
"""

from backend.rag.retrieval.bm25_sparse import BM25SparseEncoder, term_id, tokenize


class TestTokenize:

    def test_lowercases_and_drops_stopwords(self):
        assert tokenize("The Basel III Capital Ratio of banks") == [
            "basel", "iii", "capital", "ratio", "banks",
        ]

    def test_empty_text(self):
        assert tokenize("") == []
        assert tokenize("the of and") == []


class TestTermId:

    def test_stable_and_non_negative(self):
        assert term_id("liquidity") == term_id("liquidity")
        assert 0 <= term_id("liquidity") < 2**31

    def test_distinct_tokens_get_distinct_ids(self):
        assert term_id("ucits") != term_id("aifmd")


class TestEncoder:

    def test_document_vector_shape(self):
        vec = BM25SparseEncoder().embed_document("capital capital ratio")
        assert len(vec["indices"]) == 2
        assert len(vec["values"]) == 2
        assert vec["indices"] == sorted(vec["indices"])

    def test_repeated_term_weighs_more_but_saturates(self):
        enc = BM25SparseEncoder()
        once = enc.embed_document("capital ratio")
        twice = enc.embed_document("capital capital ratio")
        many = enc.embed_document(" ".join(["capital"] * 50) + " ratio")
        cid = term_id("capital")

        def weight(vec):
            return vec["values"][vec["indices"].index(cid)]

        assert weight(twice) > weight(once)
        # BM25 saturation: weight is bounded by k1 + 1
        assert weight(many) < enc.k1 + 1

    def test_longer_documents_get_lower_term_weight(self):
        enc = BM25SparseEncoder(avg_len=10)
        cid = term_id("capital")
        short = enc.embed_document("capital ratio")
        long = enc.embed_document("capital " + " ".join(f"filler{i}" for i in range(40)))
        assert (
            long["values"][long["indices"].index(cid)]
            < short["values"][short["indices"].index(cid)]
        )

    def test_query_is_binary_and_deduplicated(self):
        vec = BM25SparseEncoder().embed_query("capital capital ratio")
        assert len(vec["indices"]) == 2
        assert vec["values"] == [1.0, 1.0]

    def test_empty_document(self):
        assert BM25SparseEncoder().embed_document("") == {"indices": [], "values": []}
