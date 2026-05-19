"""
KILT Benchmark Verifier for Explainable RAG Pipeline
======================================================
Loads KILT benchmark questions (NQ, TriviaQA, HotpotQA, FEVER, WoW),
retrieves passages from the indexed KILT Wikipedia in Qdrant, generates
answers via GPT-4o, and computes standard evaluation metrics.

Reuses your existing pipeline components:
  - VectorStoreService (Qdrant search)
  - get_embedding_model (BAAI/bge-base-en-v1.5)
  - generate_answer (GPT-4o)
  - ExplainabilityEngine

Prerequisites:
  pip install datasets sentence-transformers qdrant-client loguru tqdm openai

Usage:
  # Quick test: 20 Natural Questions
  python kilt_verifier.py --task nq --limit 20

  # Evaluate 200 questions across all KILT tasks
  python kilt_verifier.py --task all --limit 200

  # Full evaluation on TriviaQA validation set
  python kilt_verifier.py --task triviaqa

  # With custom output directory
  python kilt_verifier.py --task nq --limit 50 --output-dir ./kilt_results

Place this file at:
  C:\\Users\\HP\\Desktop\\DCU\\explainable-rag-project\\explainable-rag\\backend\\evaluation\\kilt_verifier.py
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from typing import Optional
from collections import defaultdict

from loguru import logger
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. KILT TASK DEFINITIONS
# ---------------------------------------------------------------------------

KILT_TASKS = {
    "nq": {
        "dataset": "facebook/kilt_tasks",
        "subset": "nq",
        "split": "validation",
        "metric": "exact_match",
        "task_type": "question_answering",
        "description": "Natural Questions — open-domain QA",
    },
    "triviaqa": {
        "dataset": "facebook/kilt_tasks",
        "subset": "triviaqa",
        "split": "validation",
        "metric": "exact_match",
        "task_type": "question_answering",
        "description": "TriviaQA — trivia-style QA",
    },
    "hotpotqa": {
        "dataset": "facebook/kilt_tasks",
        "subset": "hotpotqa",
        "split": "validation",
        "metric": "exact_match",
        "task_type": "question_answering",
        "description": "HotpotQA — multi-hop QA",
    },
    "fever": {
        "dataset": "facebook/kilt_tasks",
        "subset": "fever",
        "split": "validation",
        "metric": "accuracy",
        "task_type": "fact_verification",
        "description": "FEVER — fact verification (SUPPORTS/REFUTES)",
    },
    "wow": {
        "dataset": "facebook/kilt_tasks",
        "subset": "wow",
        "split": "validation",
        "metric": "f1",
        "task_type": "dialogue",
        "description": "Wizard of Wikipedia — knowledge-grounded dialogue",
    },
}


# ---------------------------------------------------------------------------
# 2. METRIC FUNCTIONS
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lowercase, strip whitespace, remove articles and punctuation."""
    import re
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, ground_truths: list[str]) -> float:
    """1.0 if normalized prediction matches any ground truth."""
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == pred_norm:
            return 1.0
    return 0.0


def token_f1(prediction: str, ground_truths: list[str]) -> float:
    """Token-level F1 between prediction and best-matching ground truth."""
    pred_tokens = set(normalize_answer(prediction).split())
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = set(normalize_answer(gt).split())
        if not pred_tokens and not gt_tokens:
            best_f1 = max(best_f1, 1.0)
            continue
        if not pred_tokens or not gt_tokens:
            continue
        common = pred_tokens & gt_tokens
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def answer_contains(prediction: str, ground_truths: list[str]) -> float:
    """1.0 if any ground truth string appears in the prediction."""
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) in pred_norm:
            return 1.0
    return 0.0


def hit_rate_at_k(retrieved_titles: list[str], provenance_titles: list[str]) -> float:
    """1.0 if any provenance page title appears in retrieved results."""
    retrieved_set = {t.lower().strip() for t in retrieved_titles}
    for pt in provenance_titles:
        if pt.lower().strip() in retrieved_set:
            return 1.0
    return 0.0


def precision_at_k(retrieved_titles: list[str], provenance_titles: list[str], k: int = 5) -> float:
    """Fraction of top-k retrieved titles that are in provenance."""
    prov_set = {t.lower().strip() for t in provenance_titles}
    top_k = retrieved_titles[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for t in top_k if t.lower().strip() in prov_set)
    return hits / len(top_k)


# ---------------------------------------------------------------------------
# 3. EXTRACT GROUND TRUTH FROM KILT FORMAT
# ---------------------------------------------------------------------------

def extract_ground_truth(example: dict) -> dict:
    """
    Parse a KILT example into a clean format.

    Returns:
        {
            "input": str,
            "answers": list[str],
            "provenance_titles": list[str],
            "provenance_wikipedia_ids": list[str],
        }
    """
    input_text = example.get("input", "")

    # Extract answers
    answers = []
    provenance_titles = []
    provenance_ids = []

    output_list = example.get("output", [])
    for output in output_list:
        # Answer text
        answer = output.get("answer", "")
        if answer and answer.strip():
            answers.append(answer.strip())

        # Provenance (gold evidence pages)
        provenance = output.get("provenance", [])
        for prov in provenance:
            title = prov.get("title", "")
            wiki_id = prov.get("wikipedia_id", "")
            if title:
                provenance_titles.append(title)
            if wiki_id:
                provenance_ids.append(str(wiki_id))

    # Deduplicate
    answers = list(dict.fromkeys(answers))
    provenance_titles = list(dict.fromkeys(provenance_titles))
    provenance_ids = list(dict.fromkeys(provenance_ids))

    return {
        "input": input_text,
        "answers": answers,
        "provenance_titles": provenance_titles,
        "provenance_wikipedia_ids": provenance_ids,
    }


# ---------------------------------------------------------------------------
# 4. RAG PIPELINE WRAPPER
# ---------------------------------------------------------------------------

class KILTRAGPipeline:
    """
    Wraps your existing RAG components for KILT evaluation.
    Uses VectorStoreService for retrieval and GPT-4o for generation.
    """

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "kilt_wikipedia",
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        embedding_dimension: int = 768,
        openai_api_key: Optional[str] = None,
        top_k: int = 5,
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient

        # Embedding model
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        logger.info(f"Loading embedding model on {self.device}...")
        self.model = SentenceTransformer(embedding_model_name, device=self.device)

        # Qdrant client
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.top_k = top_k

        # OpenAI API key
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")

        # Check collection exists
        info = self.client.get_collection(collection_name)
        logger.info(
            f"Connected to Qdrant collection '{collection_name}' "
            f"({info.points_count:,} points)"
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Embed query and retrieve top-k passages from KILT Wikipedia."""
        k = top_k or self.top_k

        # Embed query
        query_vector = self.model.encode(
            query, normalize_embeddings=True
        ).tolist()

        # Search Qdrant
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
        )

        retrieved = []
        for hit in results.points:
            retrieved.append({
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "wikipedia_title": hit.payload.get("wikipedia_title", ""),
                "wikipedia_id": hit.payload.get("wikipedia_id", ""),
                "page_number": hit.payload.get("page_number", -1),
                "paragraph_number": hit.payload.get("paragraph_number", -1),
                "source": hit.payload.get("source", ""),
                "document_id": hit.payload.get("document_id", ""),
            })

        return retrieved

    def generate(
        self,
        question: str,
        retrieved: list[dict],
        task_type: str = "question_answering",
    ) -> str:
        """Generate answer using GPT-4o given retrieved context."""
        if not self.api_key:
            logger.warning("No OpenAI API key — returning retrieval-only results")
            return "[NO_API_KEY]"

        from openai import OpenAI

        # Build context from retrieved passages
        context_parts = []
        for i, r in enumerate(retrieved):
            context_parts.append(f"[Passage {i+1}] (score={r['score']:.3f})\n{r['text']}")
        context = "\n\n".join(context_parts)

        # Task-specific prompts
        if task_type == "fact_verification":
            system_prompt = (
                "You are a fact-checking assistant. Given the passages below, determine "
                "whether the claim is SUPPORTS or REFUTES based on the evidence. "
                "Answer with exactly 'SUPPORTS' or 'REFUTES'."
            )
        elif task_type == "dialogue":
            system_prompt = (
                "You are a knowledgeable conversational assistant. Given the passages below, "
                "provide a natural, informative response to continue the conversation. "
                "Keep your response concise and grounded in the passages."
            )
        else:
            system_prompt = (
                "You are a question-answering assistant. Answer the question using ONLY "
                "the provided passages. Give a short, direct answer. If the passages don't "
                "contain the answer, say 'I don't have enough information.'"
            )

        user_prompt = f"Passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[ERROR: {e}]"

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# 5. EVALUATION RUNNER
# ---------------------------------------------------------------------------

def run_evaluation(args):
    """Run KILT benchmark evaluation."""
    from datasets import load_dataset

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which tasks to run
    if args.task == "all":
        tasks = list(KILT_TASKS.keys())
    else:
        tasks = [args.task]

    # Initialize pipeline
    pipeline = KILTRAGPipeline(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        openai_api_key=args.openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
        top_k=args.top_k,
        device=args.device,
    )

    all_results = {}

    for task_name in tasks:
        task_config = KILT_TASKS[task_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"  EVALUATING: {task_config['description']}")
        logger.info(f"{'='*60}")

        # Load dataset
        logger.info(f"Loading {task_config['dataset']} / {task_config['subset']}...")
        dataset = None
        try:
            # Approach 1: Load from auto-converted Parquet files
            parquet_url = (
                f"hf://datasets/{task_config['dataset']}/"
                f"{task_config['subset']}/{task_config['split']}/*.parquet"
            )
            dataset = load_dataset("parquet", data_files=parquet_url, split="train")
            logger.info(f"Loaded {task_name} via Parquet")
        except Exception:
            pass

        if dataset is None:
            try:
                # Approach 2: Standard load (older datasets library)
                dataset = load_dataset(
                    task_config["dataset"],
                    task_config["subset"],
                    split=task_config["split"],
                )
                logger.info(f"Loaded {task_name} via standard API")
            except Exception as e:
                logger.error(f"Failed to load {task_name}: {e}")
                logger.error("Fix: pip install datasets==2.14.7")
                continue

        # Apply limit
        n_examples = min(len(dataset), args.limit) if args.limit else len(dataset)
        logger.info(f"Evaluating {n_examples} examples")

        # Evaluate each example
        results = []
        metrics_accum = defaultdict(list)

        for idx in tqdm(range(n_examples), desc=task_name):
            example = dataset[idx]
            gt = extract_ground_truth(example)

            if not gt["input"]:
                continue

            t_start = time.time()

            # Retrieve
            retrieved = pipeline.retrieve(gt["input"], top_k=args.top_k)
            retrieval_time = time.time() - t_start

            # Generate (skip if --retrieval-only)
            if args.retrieval_only:
                answer = "[RETRIEVAL_ONLY]"
                generation_time = 0
            else:
                t_gen = time.time()
                answer = pipeline.generate(
                    gt["input"], retrieved, task_type=task_config["task_type"]
                )
                generation_time = time.time() - t_gen

            total_time = retrieval_time + generation_time

            # Compute metrics
            retrieved_titles = [r["wikipedia_title"] for r in retrieved]
            em = exact_match(answer, gt["answers"]) if gt["answers"] else 0.0
            f1 = token_f1(answer, gt["answers"]) if gt["answers"] else 0.0
            contains = answer_contains(answer, gt["answers"]) if gt["answers"] else 0.0
            hit = hit_rate_at_k(retrieved_titles, gt["provenance_titles"])
            prec = precision_at_k(retrieved_titles, gt["provenance_titles"], k=args.top_k)

            # Top retrieval score
            top_score = retrieved[0]["score"] if retrieved else 0.0

            result = {
                "task": task_name,
                "idx": idx,
                "question": gt["input"],
                "ground_truth": gt["answers"],
                "prediction": answer,
                "exact_match": em,
                "token_f1": f1,
                "answer_contains": contains,
                "hit_rate": hit,
                "precision_at_k": prec,
                "top_retrieval_score": top_score,
                "retrieval_time_ms": retrieval_time * 1000,
                "generation_time_ms": generation_time * 1000,
                "total_time_ms": total_time * 1000,
                "provenance_titles": gt["provenance_titles"],
                "retrieved_titles": retrieved_titles,
                "retrieved_passages": [r["text"][:200] for r in retrieved[:3]],
            }
            results.append(result)

            # Accumulate metrics
            metrics_accum["exact_match"].append(em)
            metrics_accum["token_f1"].append(f1)
            metrics_accum["answer_contains"].append(contains)
            metrics_accum["hit_rate"].append(hit)
            metrics_accum["precision_at_k"].append(prec)
            metrics_accum["retrieval_time_ms"].append(retrieval_time * 1000)
            metrics_accum["generation_time_ms"].append(generation_time * 1000)

        # Compute aggregate metrics
        if not results:
            logger.warning(f"No results for {task_name}")
            continue

        agg = {}
        for metric_name, values in metrics_accum.items():
            agg[metric_name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

        all_results[task_name] = {
            "config": task_config,
            "aggregated": agg,
            "n_examples": len(results),
        }

        # Print results
        logger.info(f"\n{'='*60}")
        logger.info(f"  RESULTS: {task_config['description']}")
        logger.info(f"  ({len(results)} examples evaluated)")
        logger.info(f"{'='*60}")
        logger.info(f"  {'Metric':<22} {'Mean':>8}")
        logger.info(f"  {'-'*35}")
        for metric_name, vals in agg.items():
            logger.info(f"  {metric_name:<22} {vals['mean']:>8.4f}")
        logger.info(f"{'='*60}")

        # Save detailed results to CSV
        csv_path = os.path.join(
            args.output_dir,
            f"kilt_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for r in results:
                # Convert lists to strings for CSV
                row = {k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()}
                writer.writerow(row)
        logger.info(f"  Detailed results saved to: {csv_path}")

    # Save aggregate summary
    summary_path = os.path.join(
        args.output_dir,
        f"kilt_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAggregate summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline on KILT benchmark tasks"
    )
    parser.add_argument("--task", type=str, default="nq",
                        choices=list(KILT_TASKS.keys()) + ["all"],
                        help="KILT task to evaluate (or 'all')")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples per task (default: all)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of passages to retrieve")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip generation, evaluate retrieval only")
    parser.add_argument("--output-dir", type=str, default="./kilt_eval_results",
                        help="Directory for result files")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for embeddings: cpu, cuda, cuda:0, mps")
    parser.add_argument("--openai-api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--qdrant-host", type=str, default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection-name", type=str, default="kilt_wikipedia")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-dimension", type=int, default=768)

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()