"""
KILT Wikipedia Indexer for Explainable RAG Pipeline
=====================================================
Downloads the facebook/kilt_wikipedia dataset (5,903,530 articles),
preprocesses each article into 100-word passages (following Karpukhin et al.),
embeds them using BAAI/bge-base-en-v1.5, and inserts into Qdrant.

Supports:
  - Full dataset or configurable subset (--limit N)
  - Resume from last checkpoint (--resume)
  - Batch embedding + upsert for memory efficiency
  - Progress logging with ETA

Prerequisites:
  pip install datasets sentence-transformers qdrant-client loguru tqdm

Usage:
  # Index a small test subset (1000 articles)
  python kilt_indexer.py --limit 1000

  # Index 100K articles (good for evaluation)
  python kilt_indexer.py --limit 100000

  # Index the full 5.9M dataset (requires ~48+ hours on CPU, ~6 hrs on GPU)
  python kilt_indexer.py

  # Resume from where you left off after a crash
  python kilt_indexer.py --resume

  # Specify GPU device
  python kilt_indexer.py --device cuda:0

Place this file at:
  C:\\Users\\HP\\Desktop\\DCU\\explainable-rag-project\\explainable-rag\\backend\\evaluation\\kilt_indexer.py
"""

import os
import sys
import json
import time
import uuid
import argparse
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm
from qdrant_client.models import PointStruct

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULTS = {
    "qdrant_host": "localhost",
    "qdrant_port": 6333,
    "collection_name": "kilt_wikipedia",       # Separate from your UCITS collection
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "embedding_dimension": 768,
    "max_passage_words": 100,                   # Karpukhin et al. preprocessing
    "embed_batch_size": 256,                    # Passages per embedding batch
    "upsert_batch_size": 500,                   # Points per Qdrant upsert
    "checkpoint_every": 10_000,                 # Save progress every N articles
    "checkpoint_file": "kilt_index_checkpoint.json",
}


# ---------------------------------------------------------------------------
# 2. PASSAGE SPLITTING  (Karpukhin et al. preprocessing)
# ---------------------------------------------------------------------------

def split_article_to_passages(
    title: str,
    text: dict,
    max_words: int = 100,
) -> list[dict]:
    """
    Split a KILT Wikipedia article into passages of max `max_words` words.
    Each passage is prepended with the article title (following DPR convention):
        "Article Title [SEP] passage text ..."

    The KILT `text` field is a dict with:
      - "paragraph": list of paragraph strings (the actual text)
      OR it can be a single string in some formats.

    Returns a list of dicts:
        [{"text": "Title [SEP] passage...", "title": ..., "paragraph_idx": ...}, ...]
    """
    # Extract paragraph list from KILT format
    if isinstance(text, dict):
        paragraphs = text.get("paragraph", [])
    elif isinstance(text, str):
        paragraphs = [text]
    elif isinstance(text, list):
        paragraphs = text
    else:
        paragraphs = [str(text)]

    passages = []
    passage_idx = 0

    for para_idx, paragraph in enumerate(paragraphs):
        if not paragraph or not paragraph.strip():
            continue

        words = paragraph.strip().split()
        # Split paragraph into chunks of max_words
        for start in range(0, len(words), max_words):
            chunk_words = words[start : start + max_words]
            chunk_text = " ".join(chunk_words)

            # Prepend title (DPR convention: "title [SEP] passage")
            full_text = f"{title} [SEP] {chunk_text}"

            passages.append({
                "text": full_text,
                "title": title,
                "paragraph_idx": para_idx,
                "passage_idx": passage_idx,
                "word_count": len(chunk_words),
            })
            passage_idx += 1

    return passages


# ---------------------------------------------------------------------------
# 3. CHECKPOINT MANAGEMENT
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> dict:
    """Load the last checkpoint (articles processed so far)."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"articles_processed": 0, "passages_indexed": 0, "elapsed_seconds": 0}


def save_checkpoint(path: str, data: dict):
    """Save current progress."""
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# 4. MAIN INDEXING PIPELINE
# ---------------------------------------------------------------------------

def run_indexer(args):
    """Main indexing loop."""
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    # ---- Load embedding model ----
    logger.info(f"Loading embedding model: {args.embedding_model}")
    device = args.device or ("cuda" if _has_cuda() else "cpu")
    model = SentenceTransformer(args.embedding_model, device=device)
    logger.info(f"Embedding model loaded on {device}")

    # ---- Connect to Qdrant ----
    logger.info(f"Connecting to Qdrant at {args.qdrant_host}:{args.qdrant_port}")
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    client.get_collections()  # Test connection
    logger.info("Qdrant connected")

    # ---- Ensure collection exists ----
    collections = [c.name for c in client.get_collections().collections]
    if args.collection_name not in collections:
        client.create_collection(
            collection_name=args.collection_name,
            vectors_config=VectorParams(
                size=args.embedding_dimension,
                distance=Distance.COSINE,
            ),
            # Optimized for large-scale indexing
            optimizers_config={
                "indexing_threshold": 50_000,   # Delay HNSW build during bulk insert
            },
        )
        logger.info(f"Created collection '{args.collection_name}' ({args.embedding_dimension}d, cosine)")
    else:
        info = client.get_collection(args.collection_name)
        logger.info(
            f"Collection '{args.collection_name}' exists "
            f"({info.points_count} points already indexed)"
        )

    # ---- Load checkpoint ----
    checkpoint = load_checkpoint(args.checkpoint_file)
    start_from = checkpoint["articles_processed"] if args.resume else 0
    total_passages = checkpoint["passages_indexed"] if args.resume else 0
    elapsed_prev = checkpoint["elapsed_seconds"] if args.resume else 0

    if args.resume and start_from > 0:
        logger.info(f"Resuming from article {start_from} ({total_passages} passages already indexed)")

    # ---- Load KILT Wikipedia dataset (streaming to avoid 36GB RAM) ----
    logger.info("Loading facebook/kilt_wikipedia dataset (streaming)...")
    dataset = None

    # Approach 1: Load from auto-converted Parquet files on HF Hub
    # try:
    #     dataset = load_dataset(
    #         "parquet",
    #         data_files="hf://datasets/facebook/kilt_wikipedia/2019-08-01/full/*.parquet",
    #         split="train",
    #         streaming=True,
    #     )
    #     logger.info("Loaded via Parquet files from HF Hub")
    # except Exception as e1:
    #     logger.warning(f"Parquet approach failed: {e1}")

    # Approach 2: Standard load (works with older datasets library)
    if dataset is None:
        try:
            dataset = load_dataset(
                "facebook/kilt_wikipedia",
                "2019-08-01",
                split="full",
                streaming=True,
                trust_remote_code=True,
            )
            logger.info("Loaded via standard HF datasets API")
        except Exception as e2:
            logger.error(
                f"Could not load KILT Wikipedia: {e2}\n"
                "Fix: pip install datasets==2.14.7  (supports legacy loading scripts)"
            )
            sys.exit(1)

    logger.info("Dataset stream ready")

    # Determine how many articles to process
    total_articles = args.limit if args.limit else 5_903_530
    logger.info(f"Will process {total_articles:,} articles (limit={args.limit})")

    # ---- Indexing loop ----
    passage_buffer_texts = []
    passage_buffer_meta = []
    articles_done = 0
    t_start = time.time()

    for article_idx, article in enumerate(dataset):
        # Skip already-processed articles on resume
        if article_idx < start_from:
            continue

        # Check limit
        if args.limit and articles_done >= args.limit:
            break

        # Split article into 100-word passages
        title = article.get("wikipedia_title", "")
        text = article.get("text", {})
        wiki_id = article.get("wikipedia_id", "")
        kilt_id = article.get("kilt_id", "")

        passages = split_article_to_passages(title, text, max_words=args.max_passage_words)

        for p in passages:
            passage_buffer_texts.append(p["text"])
            passage_buffer_meta.append({
                "title": p["title"],
                "wikipedia_id": wiki_id,
                "kilt_id": kilt_id,
                "paragraph_idx": p["paragraph_idx"],
                "passage_idx": p["passage_idx"],
                "word_count": p["word_count"],
                "document_id": "kilt_wikipedia",
            })

        articles_done += 1

        # ---- Flush buffer when large enough ----
        if len(passage_buffer_texts) >= args.embed_batch_size:
            n_indexed = _flush_buffer(
                client, model, args.collection_name,
                passage_buffer_texts, passage_buffer_meta,
                args.embed_batch_size, args.upsert_batch_size,
            )
            total_passages += n_indexed
            passage_buffer_texts.clear()
            passage_buffer_meta.clear()

        # ---- Checkpoint ----
        if articles_done % args.checkpoint_every == 0:
            elapsed = time.time() - t_start + elapsed_prev
            save_checkpoint(args.checkpoint_file, {
                "articles_processed": article_idx + 1,
                "passages_indexed": total_passages,
                "elapsed_seconds": elapsed,
            })
            rate = articles_done / (time.time() - t_start) if (time.time() - t_start) > 0 else 0
            remaining = (total_articles - articles_done) / rate if rate > 0 else 0
            logger.info(
                f"[Checkpoint] Articles: {articles_done:,}/{total_articles:,} | "
                f"Passages indexed: {total_passages:,} | "
                f"Rate: {rate:.1f} art/s | "
                f"ETA: {remaining/3600:.1f}h"
            )

    # ---- Flush remaining buffer ----
    if passage_buffer_texts:
        n_indexed = _flush_buffer(
            client, model, args.collection_name,
            passage_buffer_texts, passage_buffer_meta,
            args.embed_batch_size, args.upsert_batch_size,
        )
        total_passages += n_indexed

    # ---- Final checkpoint ----
    elapsed = time.time() - t_start + elapsed_prev
    save_checkpoint(args.checkpoint_file, {
        "articles_processed": start_from + articles_done,
        "passages_indexed": total_passages,
        "elapsed_seconds": elapsed,
    })

    # ---- Summary ----
    info = client.get_collection(args.collection_name)
    logger.info("=" * 60)
    logger.info("  KILT WIKIPEDIA INDEXING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Articles processed:  {articles_done:,}")
    logger.info(f"  Passages indexed:    {total_passages:,}")
    logger.info(f"  Qdrant points:       {info.points_count:,}")
    logger.info(f"  Total time:          {elapsed/3600:.2f} hours")
    logger.info(f"  Collection:          {args.collection_name}")
    logger.info("=" * 60)


def _flush_buffer(
    client, model, collection_name,
    texts: list[str], metas: list[dict],
    embed_batch_size: int, upsert_batch_size: int,
) -> int:
    """Embed a buffer of passages and upsert them into Qdrant."""
    if not texts:
        return 0

    # Embed in sub-batches (sentence-transformers handles internal batching)
    vectors = model.encode(
        texts,
        batch_size=embed_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # BGE models need normalization for cosine
    ).tolist()

    # Build Qdrant points
    points = []
    for text, vector, meta in zip(texts, vectors, metas):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": text,
                    "page_number": meta["paragraph_idx"],
                    "paragraph_number": meta["passage_idx"],
                    "paragraph_numbers": [meta["passage_idx"]],
                    "paragraph_match_type": "kilt_passage",
                    "source": meta["title"],
                    "document_id": meta["document_id"],
                    "chunk_index": meta["passage_idx"],
                    # KILT-specific metadata
                    "wikipedia_id": meta["wikipedia_id"],
                    "kilt_id": meta["kilt_id"],
                    "wikipedia_title": meta["title"],
                    "word_count": meta["word_count"],
                },
            )
        )

    # Upsert in batches
    for start in range(0, len(points), upsert_batch_size):
        batch = points[start : start + upsert_batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    return len(points)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Index KILT Wikipedia into Qdrant for RAG evaluation"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of articles to index (default: all 5.9M)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for embeddings: cpu, cuda, cuda:0, mps")
    parser.add_argument("--qdrant-host", type=str, default=DEFAULTS["qdrant_host"])
    parser.add_argument("--qdrant-port", type=int, default=DEFAULTS["qdrant_port"])
    parser.add_argument("--collection-name", type=str, default=DEFAULTS["collection_name"])
    parser.add_argument("--embedding-model", type=str, default=DEFAULTS["embedding_model"])
    parser.add_argument("--embedding-dimension", type=int, default=DEFAULTS["embedding_dimension"])
    parser.add_argument("--max-passage-words", type=int, default=DEFAULTS["max_passage_words"])
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULTS["embed_batch_size"])
    parser.add_argument("--upsert-batch-size", type=int, default=DEFAULTS["upsert_batch_size"])
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULTS["checkpoint_every"])
    parser.add_argument("--checkpoint-file", type=str, default=DEFAULTS["checkpoint_file"])

    args = parser.parse_args()
    run_indexer(args)


if __name__ == "__main__":
    main()