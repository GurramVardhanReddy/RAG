"""
Dense vs hybrid (dense + BM25 -> RRF) retrieval benchmark
=========================================================
Measures what the BM25 arm changes for the domain models that have no
sparse output of their own. It runs the real VectorStoreService and
HybridRetriever against an in-memory Qdrant, so no server or API key is
needed. The reranker is off: this isolates first-stage retrieval.

Benchmarks (BEIR, test split, from the Hugging Face hub):
    scifact   → Scientific/Academic model (allenai/scibert_scivocab_uncased)
    nfcorpus  → Medical/Biomedical model (NeuML/pubmedbert-base-embeddings)
Each is also run with the General model (BAAI/bge-base-en-v1.5).

Modes:
    dense         dense arm only (what every query got before the fix)
    bm25          BM25 arm only
    hybrid        dense + BM25, RRF, shipped prefetch (dense 40 / sparse 10)
    hybrid_equal  dense + BM25, RRF, equal prefetch (40 / 40)

Metrics at k=10: nDCG, Recall, MRR, with a paired bootstrap 95% CI for
hybrid minus dense nDCG@10.

Usage:
    python scripts/eval_hybrid.py                     # both datasets, both models
    python scripts/eval_hybrid.py --datasets scifact --max-queries 100
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402
from loguru import logger  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from backend.app.services.vector_store import VectorStoreService  # noqa: E402
from backend.rag.embedding.domain_detector import DOMAIN_MODEL_MAP  # noqa: E402
from backend.rag.embedding.embedder import get_embedding_model  # noqa: E402
from backend.rag.retrieval.bm25_sparse import BM25SparseEncoder  # noqa: E402
from backend.rag.retrieval.retriever import HybridRetriever  # noqa: E402

DATASETS = {
    "scifact": "Scientific/Academic",
    "nfcorpus": "Medical/Biomedical",
}
K = 10
DOC_ID = "bench"


def load_beir(name: str, max_queries: int | None):
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{name}-qrels", split="test")

    relevant: dict[str, dict[str, int]] = {}
    for row in qrels:
        if int(row["score"]) > 0:
            relevant.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])

    qtext = {str(r["_id"]): r["text"] for r in queries}
    qids = sorted(q for q in relevant if q in qtext)
    if max_queries:
        qids = qids[:max_queries]
    docs = [(str(r["_id"]), f"{r['title']}. {r['text']}".strip(". ")) for r in corpus]
    return docs, {q: qtext[q] for q in qids}, relevant


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = K) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int = K) -> float:
    return len(set(ranked[:k]) & set(rel)) / len(rel)


def mrr_at_k(ranked: list[str], rel: dict[str, int], k: int = K) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 7):
    """95% CI and two-sided p-value for mean(b - a)."""
    diff = b - a
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(diff.mean()), float(lo), float(hi), float(min(p, 1.0))


def build_index(docs, domain: str, model_name: str):
    """Embed and store the corpus through the production VectorStoreService."""
    store = VectorStoreService()
    store.client = QdrantClient(":memory:")
    model = get_embedding_model(model_name)
    texts = [t for _, t in docs]

    t0 = time.time()
    dense = model.embed_documents(texts)
    embed_s = time.time() - t0

    chunks = [Document(page_content=t, metadata={"source": cid}) for cid, t in docs]
    store.add_documents(
        chunks=chunks,
        dense_vectors=dense,
        sparse_vectors=[{"indices": [], "values": []}] * len(texts),
        document_id=DOC_ID,
        domain=domain,
        embedding_model_name=model_name,
        bm25_vectors=BM25SparseEncoder().embed_documents(texts),
    )
    return store, embed_s


def run_mode(store, domain, model_name, queries, mode: str) -> dict[str, list[str]]:
    registry = {DOC_ID: {"domain": domain, "embedding_model": model_name}}
    sparse_prefetch = 40 if mode == "hybrid_equal" else 10
    retriever = HybridRetriever(
        vector_store=store,
        use_hybrid=mode != "dense",
        use_reranker=False,
        dense_prefetch=40,
        sparse_prefetch=sparse_prefetch,
        document_registry=registry,
    )
    bm25 = BM25SparseEncoder()
    ranked = {}
    for qid, text in queries.items():
        if mode == "bm25":
            res = store.client.query_points(
                collection_name=store._collection_name(domain),
                query=store._to_qdrant_sparse(bm25.embed_query(text)),
                using="bm25",
                limit=K,
                with_payload=True,
            )
            ranked[qid] = [p.payload["source"] for p in res.points]
        else:
            out = retriever.retrieve(text, top_k=K, document_id=DOC_ID)
            ranked[qid] = [r["source"] for r in out["results"]]
    return ranked


def evaluate(name: str, domain: str, model_name: str, max_queries: int | None) -> dict:
    docs, queries, relevant = load_beir(name, max_queries)
    print(f"\n[{name}] {len(docs)} docs, {len(queries)} queries, model={model_name}")
    store, embed_s = build_index(docs, domain, model_name)
    print(f"  indexed in {embed_s:.0f}s")

    out = {
        "dataset": name, "domain": domain, "model": model_name,
        "docs": len(docs), "queries": len(queries), "embed_seconds": round(embed_s, 1),
        "modes": {},
    }
    per_query = {}
    for mode in ("dense", "bm25", "hybrid", "hybrid_equal"):
        t0 = time.time()
        ranked = run_mode(store, domain, model_name, queries, mode)
        ms = (time.time() - t0) * 1000 / len(queries)
        scores = np.array([ndcg_at_k(ranked[q], relevant[q]) for q in queries])
        per_query[mode] = scores
        out["modes"][mode] = {
            "ndcg@10": round(float(scores.mean()), 4),
            "recall@10": round(float(np.mean([recall_at_k(ranked[q], relevant[q]) for q in queries])), 4),
            "mrr@10": round(float(np.mean([mrr_at_k(ranked[q], relevant[q]) for q in queries])), 4),
            "ms_per_query": round(ms, 1),
        }
        print(f"  {mode:13s} nDCG@10={scores.mean():.4f}  {ms:.0f} ms/query")

    for mode in ("hybrid", "hybrid_equal"):
        mean, lo, hi, p = paired_bootstrap(per_query["dense"], per_query[mode])
        out["modes"][mode]["delta_ndcg_vs_dense"] = {
            "mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "p": round(p, 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--no-general", action="store_true", help="skip the General-model runs")
    ap.add_argument("--out", default="evaluation_results/hybrid_bm25_eval.json")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    results = []
    for name in args.datasets:
        domain = DATASETS[name]
        models = [DOMAIN_MODEL_MAP[domain]]
        if not args.no_general:
            models.append(DOMAIN_MODEL_MAP["General"])
        for model_name in models:
            results.append(evaluate(name, domain, model_name, args.max_queries))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
