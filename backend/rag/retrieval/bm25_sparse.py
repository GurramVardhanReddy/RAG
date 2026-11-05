"""
BM25 Sparse Encoder — model-agnostic lexical arm for hybrid retrieval
=====================================================================
Only BGE-M3 can produce learned sparse vectors. Every domain model in
DOMAIN_MODEL_MAP is dense-only, so before this encoder existed the sparse
arm was empty and hybrid retrieval silently fell back to dense search.

This encoder gives every domain a lexical arm without loading a model:
    - Documents: BM25 term-frequency weights with length normalisation
      tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))
    - Queries:   weight 1.0 per unique term
    - IDF:       computed by Qdrant at query time (Modifier.IDF on the
      "bm25" sparse vector), so it stays correct as documents are added.

Term ids are a stable 31-bit hash of the token, so no vocabulary has to be
stored or shared between the indexing and query processes.
"""

import re
import zlib
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small English stopword list: enough to stop "the"/"of" dominating the arm
STOPWORDS = frozenset(
    "a an and are as at be been but by for from has have he her his i if in into is it its "
    "itself me my no not of on or our she so such than that the their them then there these "
    "they this those to too us was we were what when where which who whom why will with you "
    "your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def term_id(token: str) -> int:
    """Stable non-negative 31-bit id for a token (crc32 is deterministic across runs)."""
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


class BM25SparseEncoder:
    """Encode text into BM25 sparse vectors ({"indices": [...], "values": [...]})."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 256.0):
        self.k1 = k1
        self.b = b
        self.avg_len = avg_len

    def _to_sparse(self, weights: dict[int, float]) -> dict:
        indices = sorted(weights)
        return {"indices": indices, "values": [float(weights[i]) for i in indices]}

    def embed_document(self, text: str) -> dict:
        tokens = tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_len)
        weights: dict[int, float] = {}
        for token, tf in Counter(tokens).items():
            tid = term_id(token)
            # Hash collisions are rare at 31 bits; sum rather than overwrite
            weights[tid] = weights.get(tid, 0.0) + tf * (self.k1 + 1) / (tf + norm)
        return self._to_sparse(weights)

    def embed_documents(self, texts: list[str]) -> list[dict]:
        return [self.embed_document(t) for t in texts]

    def embed_query(self, text: str) -> dict:
        return self._to_sparse({term_id(t): 1.0 for t in set(tokenize(text))})
