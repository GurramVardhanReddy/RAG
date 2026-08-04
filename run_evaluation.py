"""
Explainable RAG — Evaluation Script (v3)
==========================================
Uses SELF-CONTAINED benchmarks — each dataset includes its own passages.
No need for 22M Wikipedia dumps, KILT indexing, or MS MARCO.

Benchmarks:
    1. HotpotQA (distractor) — Multi-hop reasoning
    2. SQuAD 2.0            — Reading comprehension + unanswerable
    3. BoolQ                — Yes/No fact verification
    4. FEVER (standalone)   — Claim verification with evidence
    5. CoQA                 — Conversational QA across 7 domains
    6. Custom UCITS corpus  — Domain-specific compliance QA

Usage:
    # PASS 1 — Quick run, no LLM judge (~30 min)
    python run_evaluation.py --experiment benchmarks --n 30 --no-judge

    # PASS 2 — Full run with judge on selected benchmarks (~1-2 hrs)
    python run_evaluation.py --experiment benchmarks --n 50

    # Custom corpus only (UCITS Supplement)
    python run_evaluation.py --experiment custom --n 50 --corpus data/uploads/Suppliment_document.pdf

    # Everything
    python run_evaluation.py --experiment both --n 50

    # Select specific benchmarks only
    python run_evaluation.py --experiment benchmarks --n 30 --benchmarks hotpotqa,squad,fever

References:
    - Yang et al., "HotpotQA" (EMNLP 2018)
    - Rajpurkar et al., "SQuAD 2.0" (ACL 2018)
    - Clark et al., "BoolQ" (NAACL 2019)
    - Thorne et al., "FEVER" (NAACL 2018)
    - Reddy et al., "CoQA" (TACL 2019)
    - Joshi et al., "Robust Multi-Model RAG Pipeline" (ICAAIC 2024)
    - Salemi & Zamani, "Evaluating Retrieval Quality in RAG" (SIGIR 2024)
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from loguru import logger

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.evaluation.runner import EvaluationRunner
from backend.evaluation.datasets import (
    load_hotpotqa_distractor,
    load_squad,
    load_boolq,
    load_fever,
    load_coqa,
    generate_multi_type_questions,
    save_questions,
    load_questions,
)
from backend.evaluation.metrics import compute_aggregate_metrics


# ============================================================
# BENCHMARK REGISTRY
# ============================================================

BENCHMARK_REGISTRY = {
    "hotpotqa": {
        "name": "HotpotQA_distractor",
        "loader": load_hotpotqa_distractor,
        "collection": "eval_hotpotqa",
        "description": "Multi-hop reasoning (2 gold + 8 distractor passages)",
        "reference": "Yang et al. (EMNLP 2018)",
    },
    "squad": {
        "name": "SQuAD_2.0",
        "loader": load_squad,
        "collection": "eval_squad",
        "description": "Reading comprehension + unanswerable detection",
        "reference": "Rajpurkar et al. (ACL 2018)",
    },
    "boolq": {
        "name": "BoolQ",
        "loader": load_boolq,
        "collection": "eval_boolq",
        "description": "Yes/No fact verification with passage context",
        "reference": "Clark et al. (NAACL 2019)",
    },
    "fever": {
        "name": "FEVER",
        "loader": load_fever,
        "collection": "eval_fever",
        "description": "Claim verification (SUPPORTS / REFUTES / NEI)",
        "reference": "Thorne et al. (NAACL 2018)",
    },
    "coqa": {
        "name": "CoQA",
        "loader": load_coqa,
        "collection": "eval_coqa",
        "description": "Conversational QA across 7 domains",
        "reference": "Reddy et al. (TACL 2019)",
    },
}


# ============================================================
# SINGLE BENCHMARK RUNNER
# ============================================================

def run_single_benchmark(runner, name, load_fn, n, collection_name, offset=0, batch_label=None):
    """
    Run evaluation on a single self-contained benchmark.
    
    1. Load dataset (questions + bundled passages)
    2. Index passages into a temporary Qdrant collection
    3. Run the full RAG pipeline on each question
    4. Compute metrics and save results
    5. Clean up the temporary collection

    Args:
        offset: Skip this many questions before loading `n` (for batched runs).
        batch_label: Optional label appended to the experiment name/output files
                     (e.g. "batch1_q0-25") so repeated batches don't overwrite results.
    """
    label_suffix = f" [{batch_label}]" if batch_label else ""
    logger.info(f"\n{'='*60}")
    logger.info(f"  BENCHMARK: {name}{label_suffix}")
    logger.info(f"{'='*60}")

    # Load dataset (supports offset for batch-wise runs)
    data = load_fn(n=n, offset=offset) if offset else load_fn(n=n)
    questions = data["questions"]
    corpus = data["corpus"]

    if not questions:
        logger.warning(f"No questions loaded for {name} — skipping")
        return {"aggregated_metrics": {}, "per_query_results": []}

    if not corpus:
        logger.warning(f"No corpus passages for {name} — skipping")
        return {"aggregated_metrics": {}, "per_query_results": []}

    logger.info(f"Loaded {len(questions)} questions, {len(corpus)} passages")

    # Switch to temporary collection prefix for this benchmark
    original_prefix = runner.vector_store.collection_prefix
    runner._set_eval_prefix(collection_name)

    # Clean up any leftover collection from previous runs
    runner._cleanup_eval_collection(collection_name)

    # Index passages
    logger.info(f"Indexing {len(corpus)} passages into '{collection_name}__general'...")
    corpus_texts = [p["text"] for p in corpus]
    runner.index_corpus_from_texts(texts=corpus_texts, document_id=collection_name)
    logger.info(f"Corpus indexed: {len(corpus_texts)} passages")

    # Run evaluation
    exp_name = f"benchmark_{name}" + (f"_{batch_label}" if batch_label else "")
    results = runner.run_experiment(
        questions=questions,
        document_id=collection_name,
        experiment_name=exp_name,
        top_k=5,
    )

    # Cleanup temporary collection
    runner._cleanup_eval_collection(collection_name)

    # Restore original prefix
    runner._set_eval_prefix(original_prefix)
    return results


# ============================================================
# STANDARD BENCHMARKS (ALL 5)
# ============================================================

def run_standard_benchmarks(runner, n=50, selected_benchmarks=None, offset=0, batch_label=None):
    """
    Run all (or selected) self-contained benchmarks.
    
    Args:
        runner: EvaluationRunner instance
        n: Number of questions per benchmark (per batch, if batching)
        selected_benchmarks: List of benchmark keys to run (e.g., ["hotpotqa", "fever"]).
                            If None, runs all benchmarks.
        offset: Skip this many questions before starting (for batch-wise runs).
        batch_label: Optional label for this batch, used in output filenames.
    """
    all_results = {}

    benchmarks_to_run = selected_benchmarks or list(BENCHMARK_REGISTRY.keys())

    logger.info(f"\nBenchmarks to run: {benchmarks_to_run}")
    logger.info(f"Questions per benchmark: {n} (offset={offset})")
    if batch_label:
        logger.info(f"Batch label: {batch_label}")

    for key in benchmarks_to_run:
        if key not in BENCHMARK_REGISTRY:
            logger.warning(f"Unknown benchmark '{key}' — skipping. Available: {list(BENCHMARK_REGISTRY.keys())}")
            continue

        info = BENCHMARK_REGISTRY[key]
        logger.info(f"\n--- {info['name']}: {info['description']} ---")
        logger.info(f"    Reference: {info['reference']}")

        try:
            result = run_single_benchmark(
                runner=runner,
                name=info["name"],
                load_fn=info["loader"],
                n=n,
                collection_name=info["collection"],
                offset=offset,
                batch_label=batch_label,
            )
            all_results[key] = result
        except Exception as e:
            logger.error(f"Benchmark '{key}' failed: {e}")
            all_results[key] = {
                "aggregated_metrics": {},
                "per_query_results": [],
                "error": str(e),
            }

    return all_results


# ============================================================
# CUSTOM CORPUS EVALUATION (UCITS)
# ============================================================

def run_custom_corpus_evaluation(runner, corpus_path, n_per_type=50, questions_file=None):
    """
    Run evaluation on a custom corpus (e.g., UCITS Supplement).
    Generates 4 types of questions using GPT-4o, then evaluates.
    
    Following Joshi et al. (ICAAIC 2024) multi-type evaluation methodology:
    - Short-form QA
    - Long-form QA
    - True/False verification
    - Multiple Choice (MCQ)
    """
    from backend.app.core.config import get_settings
    settings = get_settings()

    # Switch to custom corpus collection
    original_prefix = runner.vector_store.collection_prefix
    runner._set_eval_prefix("eval_custom")

    logger.info(f"\nIndexing corpus: {corpus_path}")
    chunks = runner.index_uploaded_document(corpus_path, document_id="custom_corpus")
    chunk_texts = [{"text": c.page_content, "page": c.metadata.get("page", 0)} for c in chunks]

    if questions_file and os.path.exists(questions_file):
        logger.info(f"Loading existing questions from {questions_file}")
        all_questions = load_questions(questions_file)
    else:
        logger.info(f"Generating {n_per_type} questions per type using GPT-4o...")
        all_questions = generate_multi_type_questions(
            chunks=chunk_texts,
            api_key=settings.openai_api_key,
            n_per_type=n_per_type,
        )
        save_path = os.path.join(runner.output_dir, "generated_questions.json")
        save_questions(all_questions, save_path)

    all_results = {}
    type_labels = {
        "short_form": "Short-Form QA",
        "long_form": "Long-Form QA",
        "true_false": "True/False Verification",
        "mcq": "Multiple Choice (MCQ)",
    }

    for qtype, label in type_labels.items():
        questions = all_questions.get(qtype, [])
        if not questions:
            logger.warning(f"No {qtype} questions generated, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"  CUSTOM: {label} ({len(questions)} questions)")
        logger.info(f"{'='*60}")

        normalized = []
        for q in questions:
            normalized.append({
                "question": q.get("question", q.get("statement", "")),
                "ground_truth": q.get("answers", [q.get("answer", "")]),
                "dataset": "custom",
                "type": qtype,
            })

        results = runner.run_experiment(
            questions=normalized,
            document_id="custom_corpus",
            experiment_name=f"custom_{qtype}",
            top_k=5,
        )
        all_results[qtype] = results

    # Cleanup custom corpus collection and restore prefix
    runner._cleanup_eval_collection("eval_custom")
    runner._set_eval_prefix(original_prefix)

    return all_results


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(all_results):
    """Print a comprehensive final summary table of all experiments."""
    logger.info("\n" + "=" * 90)
    logger.info("  FINAL SUMMARY — ALL EXPERIMENTS")
    logger.info("=" * 90)
    logger.info(
        f"{'Dataset':<28} {'Hit Rate':>9} {'MRR':>9} {'EM':>9} "
        f"{'F1':>9} {'Faith.':>9} {'Relev.':>9} {'Correct.':>9}"
    )
    logger.info("-" * 90)

    for name, result in all_results.items():
        if "error" in result:
            logger.info(f"{name:<28} {'ERROR':>9}")
            continue

        agg = result.get("aggregated_metrics", {})
        hr = agg.get("hit_rate", {}).get("mean", 0)
        mrr = agg.get("mrr", {}).get("mean", 0)
        em = agg.get("exact_match", {}).get("mean", 0)
        f1 = agg.get("f1", {}).get("mean", 0)
        faith = agg.get("faithfulness", {}).get("mean", 0)
        relev = agg.get("relevancy", {}).get("mean", 0)
        correct = agg.get("correctness", {}).get("mean", 0)
        logger.info(
            f"{name:<28} {hr:>9.4f} {mrr:>9.4f} {em:>9.4f} "
            f"{f1:>9.4f} {faith:>9.4f} {relev:>9.4f} {correct:>9.4f}"
        )

    logger.info("=" * 90)

    # Also print a quick reference for thesis
    logger.info("\n--- Quick Reference (copy for thesis) ---")
    for name, result in all_results.items():
        if "error" in result:
            continue
        agg = result.get("aggregated_metrics", {})
        n_questions = result.get("total_questions", result.get("successful", "?"))
        logger.info(f"{name}: n={n_questions}, "
                     f"HR={agg.get('hit_rate', {}).get('mean', 0):.3f}, "
                     f"MRR={agg.get('mrr', {}).get('mean', 0):.3f}, "
                     f"EM={agg.get('exact_match', {}).get('mean', 0):.3f}, "
                     f"F1={agg.get('f1', {}).get('mean', 0):.3f}")


def save_combined_results(all_results, output_dir):
    """Save a combined JSON with all experiment results."""
    combined = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
    }

    for name, result in all_results.items():
        combined["experiments"][name] = {
            "aggregated_metrics": result.get("aggregated_metrics", {}),
            "total_questions": result.get("total_questions", 0),
            "successful": result.get("successful", 0),
        }

    filepath = os.path.join(output_dir, f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info(f"\nCombined results saved to {filepath}")
    return filepath


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Explainable RAG — Pipeline Evaluation (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick pass — no judge, 30 questions per benchmark (~30 min)
  python run_evaluation.py --experiment benchmarks --n 30 --no-judge

  # Full pass — with judge, 50 questions (~1.5-2 hrs)
  python run_evaluation.py --experiment benchmarks --n 50

  # Only specific benchmarks
  python run_evaluation.py --experiment benchmarks --n 30 --benchmarks hotpotqa,fever,coqa

  # Custom UCITS corpus only
  python run_evaluation.py --experiment custom --n 50 --corpus data/uploads/Suppliment_document.pdf

  # Everything (benchmarks + custom)
  python run_evaluation.py --experiment both --n 50

  # BATCH-WISE: run questions 0-24 today, 25-49 tomorrow (same benchmark)
  python run_evaluation.py --experiment benchmarks --n 25 --offset 0  --batch-label batch1 --benchmarks hotpotqa,squad
  python run_evaluation.py --experiment benchmarks --n 25 --offset 25 --batch-label batch2 --benchmarks hotpotqa,squad
        """
    )

    parser.add_argument(
        "--experiment",
        choices=["benchmarks", "custom", "both"],
        default="both",
        help="Which experiments to run",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Questions per dataset/type, PER BATCH if using --offset (default: 50)",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip this many questions before loading --n (for batch-wise runs). "
             "E.g. --n 25 --offset 25 loads questions 26-50.",
    )
    parser.add_argument(
        "--batch-label", type=str, default=None,
        help="Label for this batch (e.g. 'batch1'). Appended to output filenames "
             "so repeated batches don't overwrite each other's results.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str, default=None,
        help="Comma-separated benchmark keys to run (default: all). "
             f"Available: {','.join(BENCHMARK_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--corpus", type=str, default=None,
        help="Path to PDF/DOCX for custom corpus evaluation",
    )
    parser.add_argument(
        "--questions-file", type=str, default=None,
        help="Path to pre-generated questions JSON (skip GPT-4o generation)",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM-as-judge evaluation (faster, cheaper)",
    )
    parser.add_argument(
        "--no-perturb", action="store_true",
        help="Skip perturbation testing (faster, avoids extra GPT-4o calls)",
    )
    parser.add_argument(
        "--perturb-delay", type=float, default=2.0,
        help="Seconds to wait between perturbation API calls (default: 2.0). "
             "Increase if hitting 429 rate limits.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./evaluation_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Parse selected benchmarks
    selected_benchmarks = None
    if args.benchmarks:
        selected_benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    # Print configuration
    logger.info("=" * 60)
    logger.info("  Explainable RAG — Pipeline Evaluation (v3)")
    logger.info("=" * 60)
    logger.info(f"  Experiment:    {args.experiment}")
    logger.info(f"  Questions/set: {args.n}")
    if args.offset:
        logger.info(f"  Offset:        {args.offset} (questions {args.offset+1}-{args.offset+args.n})")
    if args.batch_label:
        logger.info(f"  Batch label:   {args.batch_label}")
    logger.info(f"  LLM-as-judge:  {'OFF' if args.no_judge else 'ON'}")
    logger.info(f"  Benchmarks:    {selected_benchmarks or 'ALL'}")
    logger.info(f"  Output:        {args.output_dir}")
    logger.info("=" * 60)

    start_total = time.time()

    # Initialize runner
    runner = EvaluationRunner(
        output_dir=args.output_dir,
        use_llm_judge=not args.no_judge,
        collection_name="eval_benchmark",
        run_perturbation=not args.no_perturb,
        perturbation_delay=args.perturb_delay,
    )

    all_results = {}

    # ── Experiment 1: Standard Benchmarks ──
    if args.experiment in ("benchmarks", "both"):
        logger.info("\n" + ">" * 60)
        logger.info(">>> EXPERIMENT 1: Self-Contained Benchmarks")
        benchmarks_label = selected_benchmarks or list(BENCHMARK_REGISTRY.keys())
        logger.info(f">>> Running: {', '.join(benchmarks_label)}")
        logger.info(">" * 60)

        benchmark_results = run_standard_benchmarks(
            runner, n=args.n, selected_benchmarks=selected_benchmarks,
            offset=args.offset, batch_label=args.batch_label,
        )
        all_results.update(benchmark_results)

    # ── Experiment 2: Custom Corpus ──
    if args.experiment in ("custom", "both"):
        corpus_path = args.corpus
        if not corpus_path:
            # Auto-detect UCITS Supplement
            for p in [
                "data/uploads/Suppliment_document.pdf",
                "../Suppliment_document.pdf",
                "Suppliment_document.pdf",
            ]:
                if os.path.exists(p):
                    corpus_path = p
                    break

        if corpus_path and os.path.exists(corpus_path):
            logger.info("\n" + ">" * 60)
            logger.info(f">>> EXPERIMENT 2: Custom Corpus ({corpus_path})")
            logger.info(">" * 60)

            custom_results = run_custom_corpus_evaluation(
                runner=runner,
                corpus_path=corpus_path,
                n_per_type=args.n,
                questions_file=args.questions_file,
            )
            all_results.update(custom_results)
        else:
            logger.warning(
                "No corpus file found for Experiment 2. "
                "Use --corpus to specify path, or place Suppliment_document.pdf in data/uploads/"
            )

    # ── Final Summary ──
    elapsed = time.time() - start_total
    minutes = elapsed / 60

    print_final_summary(all_results)
    combined_path = save_combined_results(all_results, args.output_dir)

    logger.info(f"\nTotal evaluation time: {minutes:.1f} minutes ({elapsed:.0f}s)")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Combined summary: {combined_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()