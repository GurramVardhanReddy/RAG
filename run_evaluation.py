"""
Explainable RAG — Evaluation Script (v2)
Uses SELF-CONTAINED benchmarks — each dataset includes its own passages.
No need for 22M Wikipedia dumps.

Usage:
    python run_evaluation.py --experiment benchmarks --n 50
    python run_evaluation.py --experiment benchmarks --n 5 --no-judge
    python run_evaluation.py --experiment both --n 50 --corpus data/uploads/Suppliment_document.pdf
"""

import os, sys, json, argparse, time
from datetime import datetime
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.evaluation.runner import EvaluationRunner
from backend.evaluation.datasets import (
    load_hotpotqa_distractor, load_squad, load_boolq,
    generate_multi_type_questions, save_questions, load_questions,
)
from backend.evaluation.metrics import compute_aggregate_metrics


def run_single_benchmark(runner, name, load_fn, n, collection_name):
    logger.info(f"\n{'='*60}")
    logger.info(f"  BENCHMARK: {name}")
    logger.info(f"{'='*60}")

    data = load_fn(n=n)
    questions = data["questions"]
    corpus = data["corpus"]

    logger.info(f"Indexing {len(corpus)} passages into '{collection_name}'...")
    original_collection = runner.vector_store.collection_name
    runner.vector_store.collection_name = collection_name

    try:
        runner.vector_store.client.delete_collection(collection_name)
    except Exception:
        pass

    corpus_texts = [p["text"] for p in corpus]
    runner.index_corpus_from_texts(texts=corpus_texts, document_id=collection_name)
    logger.info(f"Corpus indexed: {len(corpus_texts)} passages")

    results = runner.run_experiment(
        questions=questions, document_id=collection_name,
        experiment_name=f"benchmark_{name}", top_k=5,
    )

    try:
        runner.vector_store.client.delete_collection(collection_name)
        logger.info(f"Cleaned up collection '{collection_name}'")
    except Exception:
        pass

    runner.vector_store.collection_name = original_collection
    return results


def run_standard_benchmarks(runner, n=50):
    all_results = {}
    all_results["hotpotqa_distractor"] = run_single_benchmark(runner, "HotpotQA_distractor", load_hotpotqa_distractor, n, "eval_hotpotqa")
    all_results["squad"] = run_single_benchmark(runner, "SQuAD_2.0", load_squad, n, "eval_squad")
    all_results["boolq"] = run_single_benchmark(runner, "BoolQ", load_boolq, n, "eval_boolq")
    return all_results


def run_custom_corpus_evaluation(runner, corpus_path, n_per_type=50, questions_file=None):
    from backend.app.core.config import get_settings
    settings = get_settings()

    logger.info(f"\nIndexing corpus: {corpus_path}")
    chunks = runner.index_uploaded_document(corpus_path, document_id="custom_corpus")
    chunk_texts = [{"text": c.page_content, "page": c.metadata.get("page", 0)} for c in chunks]

    if questions_file and os.path.exists(questions_file):
        logger.info(f"Loading existing questions from {questions_file}")
        all_questions = load_questions(questions_file)
    else:
        logger.info(f"Generating {n_per_type} questions per type using GPT-4o...")
        all_questions = generate_multi_type_questions(chunks=chunk_texts, api_key=settings.openai_api_key, n_per_type=n_per_type)
        save_path = os.path.join(runner.output_dir, "generated_questions.json")
        save_questions(all_questions, save_path)

    all_results = {}
    type_labels = {"short_form": "Short-Form QA", "long_form": "Long-Form QA", "true_false": "True/False Verification", "mcq": "Multiple Choice (MCQ)"}

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
            normalized.append({"question": q.get("question", q.get("statement", "")), "ground_truth": q.get("answers", [q.get("answer", "")]), "dataset": "custom", "type": qtype})
        results = runner.run_experiment(questions=normalized, document_id="custom_corpus", experiment_name=f"custom_{qtype}", top_k=5)
        all_results[qtype] = results

    return all_results


def print_final_summary(all_results):
    logger.info("\n" + "="*80)
    logger.info("  FINAL SUMMARY — ALL EXPERIMENTS")
    logger.info("="*80)
    logger.info(f"{'Dataset':<25} {'Hit Rate':>9} {'MRR':>9} {'EM':>9} {'F1':>9} {'Faith.':>9} {'Relev.':>9}")
    logger.info("-"*80)
    for name, result in all_results.items():
        agg = result.get("aggregated_metrics", {})
        hr = agg.get("hit_rate", {}).get("mean", 0)
        mrr = agg.get("mrr", {}).get("mean", 0)
        em = agg.get("exact_match", {}).get("mean", 0)
        f1 = agg.get("f1", {}).get("mean", 0)
        faith = agg.get("faithfulness", {}).get("mean", 0)
        relev = agg.get("relevancy", {}).get("mean", 0)
        logger.info(f"{name:<25} {hr:>9.4f} {mrr:>9.4f} {em:>9.4f} {f1:>9.4f} {faith:>9.4f} {relev:>9.4f}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Explainable RAG pipeline")
    parser.add_argument("--experiment", choices=["benchmarks", "custom", "both"], default="both")
    parser.add_argument("--n", type=int, default=50, help="Questions per dataset/type (default: 50)")
    parser.add_argument("--corpus", type=str, default=None, help="Path to PDF/DOCX for custom corpus evaluation")
    parser.add_argument("--questions-file", type=str, default=None, help="Path to pre-generated questions JSON")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge evaluation (faster, cheaper)")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", help="Directory to save results")

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("  Explainable RAG — Pipeline Evaluation (v2)")
    logger.info(f"  Experiment: {args.experiment}")
    logger.info(f"  Questions per dataset: {args.n}")
    logger.info(f"  LLM-as-judge: {'OFF' if args.no_judge else 'ON'}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("="*60)

    start_total = time.time()
    runner = EvaluationRunner(output_dir=args.output_dir, use_llm_judge=not args.no_judge, collection_name="eval_default")
    all_results = {}

    if args.experiment in ("benchmarks", "both"):
        logger.info("\n>>> EXPERIMENT 1: Self-Contained Benchmarks")
        logger.info(">>> (HotpotQA distractor, SQuAD 2.0, BoolQ)\n")
        benchmark_results = run_standard_benchmarks(runner, n=args.n)
        all_results.update(benchmark_results)

    if args.experiment in ("custom", "both"):
        corpus_path = args.corpus
        if not corpus_path:
            for p in ["data/uploads/Suppliment_document.pdf", "../Suppliment_document.pdf"]:
                if os.path.exists(p):
                    corpus_path = p
                    break
        if corpus_path and os.path.exists(corpus_path):
            logger.info(f"\n>>> EXPERIMENT 2: Custom Corpus ({corpus_path})")
            custom_results = run_custom_corpus_evaluation(runner=runner, corpus_path=corpus_path, n_per_type=args.n, questions_file=args.questions_file)
            all_results.update(custom_results)
        else:
            logger.warning("No corpus file found for Experiment 2. Use --corpus path/to/document.pdf")

    if all_results:
        print_final_summary(all_results)
        combined_path = os.path.join(args.output_dir, f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(combined_path, "w") as f:
            json.dump({name: r.get("aggregated_metrics", {}) for name, r in all_results.items()}, f, indent=2, default=str)
        logger.info(f"\nCombined results saved to {combined_path}")

    elapsed = time.time() - start_total
    logger.info(f"\nTotal evaluation time: {elapsed / 60:.1f} minutes")
    logger.info("Done!")


if __name__ == "__main__":
    main()