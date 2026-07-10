"""
Evaluation Runner for Explainable RAG Pipeline
=================================================
Runs end-to-end evaluation on standard benchmarks and custom questions.

Key features:
    - Full Layer 1 XAI metrics per query
    - LIME on retrieval (word importance)
    - Perturbation testing on generation (chunk contribution)
    - Domain-aware VectorStoreService (collection_prefix)
    - Batch-wise evaluation with offset support
"""

import os
import json
import time
import csv
from datetime import datetime
from typing import Optional, Callable
from loguru import logger

from backend.app.services.vector_store import VectorStoreService
from backend.app.core.config import get_settings
from backend.rag.embedding.embedder import get_embedding_model
from backend.rag.generation.generator import generate_answer, build_context
from backend.rag.explainability.engine import ExplainabilityEngine

from backend.evaluation.metrics import (
    exact_match,
    token_f1,
    rouge_l,
    answer_contains_ground_truth,
    hit_rate,
    mean_reciprocal_rank,
    precision_at_k,
    llm_judge_scores,
    compute_aggregate_metrics,
)


class EvaluationRunner:
    """
    Runs evaluation experiments on the RAG pipeline.

    Supports:
    - Experiment 1: Standard benchmarks (HotpotQA, SQuAD, BoolQ, FEVER, CoQA)
    - Experiment 2: Multi-type questions on custom corpus (UCITS)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "eval_benchmark",
        output_dir: str = "./evaluation_results",
        use_llm_judge: bool = True,
        run_perturbation: bool = True,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.output_dir = output_dir
        self.use_llm_judge = use_llm_judge
        self.run_perturbation = run_perturbation
        self.eval_domain = "General"
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Initializing evaluation pipeline...")
        self.embedding_model = get_embedding_model(settings.embedding_model)

        self.vector_store = VectorStoreService(
            host=qdrant_host,
            port=qdrant_port,
            collection_prefix=collection_name,
            embedding_dimension=settings.embedding_dimension,
        )
        self.vector_store.connect()

        self.explainability_engine = ExplainabilityEngine()
        logger.info("Evaluation pipeline ready")

    # ============================================================
    # COLLECTION MANAGEMENT
    # ============================================================

    def _set_eval_prefix(self, prefix: str):
        self.vector_store.collection_prefix = prefix
        self.vector_store._created_collections = set()

    def _cleanup_eval_collection(self, prefix: str):
        try:
            collection_name = f"{prefix}__general"
            self.vector_store.client.delete_collection(collection_name)
            logger.info(f"Cleaned up collection '{collection_name}'")
        except Exception:
            pass

    # ============================================================
    # CORPUS INDEXING
    # ============================================================

    def index_corpus_from_texts(
        self,
        texts: list[str],
        document_id: str = "eval_corpus",
        metadatas: Optional[list[dict]] = None,
    ):
        import numpy as np
        from langchain_core.documents import Document

        self.vector_store.ensure_collection_for_domain(self.eval_domain)

        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            meta.setdefault("page", i // 10)
            meta.setdefault("source", f"passage_{i}")
            docs.append(Document(page_content=text, metadata=meta))

        logger.info(f"Embedding {len(texts)} passages...")
        vectors = self.embedding_model.embed_documents(texts)

        class _EmptySparse:
            indices = np.array([], dtype=np.int64)
            values = np.array([], dtype=np.float32)

        dummy_sparse = [_EmptySparse() for _ in range(len(texts))]

        self.vector_store.add_documents(
            chunks=docs,
            dense_vectors=vectors,
            sparse_vectors=dummy_sparse,
            document_id=document_id,
            page_paragraphs={},
            domain=self.eval_domain,
            embedding_model_name="BAAI/bge-base-en-v1.5",
        )
        logger.info(f"Indexed {len(texts)} passages as '{document_id}'")

    def index_uploaded_document(self, file_path: str, document_id: str = "eval_doc"):
        import numpy as np
        from backend.app.services.document_loader import load_document
        from backend.rag.chunking.chunker import semantic_chunk

        logger.info(f"Loading and indexing: {file_path}")
        docs = load_document(file_path)

        page_paragraphs = {}
        for doc in docs:
            page_num = doc.metadata.get("page", 0)
            paragraphs = doc.metadata.get("paragraphs", [])
            page_paragraphs[page_num] = paragraphs

        chunks = semantic_chunk(docs, embedding_model=self.embedding_model)
        chunk_texts = [c.page_content for c in chunks]
        vectors = self.embedding_model.embed_documents(chunk_texts)

        class _EmptySparse:
            indices = np.array([], dtype=np.int64)
            values = np.array([], dtype=np.float32)

        dummy_sparse = [_EmptySparse() for _ in range(len(chunk_texts))]

        self.vector_store.ensure_collection_for_domain(self.eval_domain)
        self.vector_store.add_documents(
            chunks=chunks,
            dense_vectors=vectors,
            sparse_vectors=dummy_sparse,
            document_id=document_id,
            page_paragraphs=page_paragraphs,
            domain=self.eval_domain,
            embedding_model_name="BAAI/bge-base-en-v1.5",
        )
        logger.info(f"Indexed {len(chunks)} chunks from {file_path}")
        return chunks

    # ============================================================
    # PERTURBATION TESTING ON GENERATION
    # ============================================================

    def _run_perturbation_test(
        self,
        question: str,
        search_results: list[dict],
        original_answer: str,
        prompt_mode: str,
    ) -> dict:
        """
        For each retrieved chunk, remove it from context and re-generate.
        If the answer changes significantly, that chunk was contributing.

        Returns:
            - per_chunk_results: list of {rank, score, text_preview,
                                  answer_changed, change_ratio, contributed}
            - contributing_chunks: top chunks that actually drove the answer
            - perturbation_summary: high-level stats
        """
        # Build a generate_fn closure for the engine
        def _generate_fn(q: str, results: list[dict]) -> str:
            return generate_answer(
                question=q,
                search_results=results,
                model_name="gpt-4o",
                temperature=0,
                api_key=self.api_key,
                prompt_mode=prompt_mode,
            )

        # Use the engine's built-in perturbation test
        raw_results = self.explainability_engine.run_perturbation_test(
            question=question,
            search_results=search_results,
            original_answer=original_answer,
            generate_fn=_generate_fn,
        )

        # Enrich with chunk text and rank
        per_chunk = []
        for i, r in enumerate(raw_results):
            chunk_idx = r.get("chunk_index", i)
            chunk = search_results[chunk_idx] if chunk_idx < len(search_results) else {}
            change_ratio = r.get("answer_change_ratio", 0.0) or 0.0
            contributed = r.get("is_faithful", False)

            per_chunk.append({
                "rank": chunk_idx + 1,
                "score": round(chunk.get("score", 0.0), 4),
                "text_preview": chunk.get("text", "")[:150].strip(),
                "answer_changed": contributed,
                "change_ratio": round(change_ratio, 4),
                "contributed": contributed,
            })

        # Sort by change_ratio descending — most impactful chunks first
        sorted_chunks = sorted(per_chunk, key=lambda x: x["change_ratio"], reverse=True)

        # Contributing chunks = those where removing them changed the answer
        contributing = [c for c in sorted_chunks if c["contributed"]]
        non_contributing = [c for c in sorted_chunks if not c["contributed"]]

        summary = {
            "total_chunks_tested": len(per_chunk),
            "chunks_that_contributed": len(contributing),
            "chunks_not_used": len(non_contributing),
            "top_contributing_chunk_rank": contributing[0]["rank"] if contributing else None,
            "avg_change_ratio": round(
                sum(c["change_ratio"] for c in per_chunk) / len(per_chunk), 4
            ) if per_chunk else 0.0,
        }

        return {
            "per_chunk_results": per_chunk,
            "contributing_chunks": contributing,
            "non_contributing_chunks": non_contributing,
            "perturbation_summary": summary,
        }

    # ============================================================
    # SINGLE QUERY EVALUATION
    # ============================================================

    def evaluate_single_query(
        self,
        question: str,
        ground_truth: list[str],
        document_id: str,
        top_k: int = 5,
        metadata: Optional[dict] = None,
        prompt_mode: str = "domain",
    ) -> dict:
        """
        Run the full pipeline on one question and compute all metrics.
        Includes: retrieval metrics, generation metrics, Layer 1 XAI,
        LIME on retrieval, and perturbation testing on generation.
        """
        start_time = time.perf_counter()

        # Auto-detect prompt mode from dataset
        if metadata and prompt_mode == "domain":
            q_dataset = metadata.get("dataset", "")
            if q_dataset == "fever":
                prompt_mode = "verification"
            elif q_dataset == "boolq":
                prompt_mode = "boolq"
            elif q_dataset in ("nq", "hotpotqa", "triviaqa", "hotpotqa_distractor",
                               "squad", "coqa"):
                prompt_mode = "general"
            elif q_dataset == "custom":
                prompt_mode = "general"

        # 1. Embed query
        query_vector = self.embedding_model.embed_query(question)

        # 2. Retrieve
        search_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
            domain=self.eval_domain,
        )
        retrieved_texts = [r["text"] for r in search_results]
        scores = [r["score"] for r in search_results]

        # 3. Generate answer
        answer = generate_answer(
            question=question,
            search_results=search_results,
            model_name="gpt-4o",
            temperature=0,
            api_key=self.api_key,
            prompt_mode=prompt_mode,
        )

        # 4. Layer 1 XAI
        xai = self.explainability_engine.analyze(
            question=question,
            answer=answer,
            search_results=search_results,
            retrieval_metadata={"retrieval_method": "dense_eval"},
            total_chunks=len(search_results),
        )
        l1 = xai if hasattr(xai, "source_attribution") else None

        if l1:
            sa = xai.source_attribution
            pc = xai.provenance_completeness
            cc = xai.confidence_calibration
            ef = xai.explanation_faithfulness
        else:
            conf = self.explainability_engine.compute_confidence_score(
                scores, answer, question
            )
            sa  = type("SA",  (), {"precision": 0.0, "recall": 0.0, "f1_score": 0.0})()
            pc  = type("PC",  (), {"completeness_score": 0.0, "granularity_score": 0.0})()
            cc  = type("CC",  (), {"confidence_score": conf, "confidence_tier": "unknown"})()
            ef  = type("EF",  (), {"overall_faithfulness": 0.0, "answer_grounding_ratio": 0.0})()

        # 5. LIME on retrieval — word importance
        def _retrieve_fn(perturbed_query: str, k: int) -> list[dict]:
            pq_vector = self.embedding_model.embed_query(perturbed_query)
            return self.vector_store.search(
                query_vector=pq_vector,
                top_k=k,
                document_id=document_id,
                domain=self.eval_domain,
            )

        lime_result = self.explainability_engine.run_lime_on_query(
            query=question,
            original_results=search_results,
            retrieve_fn=_retrieve_fn,
            top_k=top_k,
        )

        # 6. Perturbation testing on generation — which chunks drove the answer?
        perturbation_result = {}
        if self.run_perturbation and len(search_results) > 1:
            try:
                perturbation_result = self._run_perturbation_test(
                    question=question,
                    search_results=search_results,
                    original_answer=answer,
                    prompt_mode=prompt_mode,
                )
            except Exception as e:
                logger.warning(f"Perturbation test failed: {e}")
                perturbation_result = {"error": str(e)}
        elif len(search_results) <= 1:
            perturbation_result = {
                "perturbation_summary": {"note": "Skipped — only 1 chunk retrieved"},
                "contributing_chunks": [],
                "per_chunk_results": [],
            }

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # 7. Retrieval metrics
        hr     = hit_rate(retrieved_texts, ground_truth)
        mrr    = mean_reciprocal_rank(retrieved_texts, ground_truth)
        p_at_k = precision_at_k(retrieved_texts, ground_truth, k=top_k)

        # 8. Generation metrics
        em       = exact_match(answer, ground_truth)
        f1       = token_f1(answer, ground_truth)
        rl       = rouge_l(answer, ground_truth)
        contains = answer_contains_ground_truth(answer, ground_truth)

        # 9. LLM-as-judge (optional)
        judge = {"faithfulness": 0.0, "relevancy": 0.0, "correctness": 0.0}
        if self.use_llm_judge:
            judge = llm_judge_scores(
                question=question,
                answer=answer,
                contexts=retrieved_texts,
                ground_truth=ground_truth[0] if ground_truth else "",
                api_key=self.api_key,
            )

        # Extract perturbation summary for flat CSV columns
        p_summary = perturbation_result.get("perturbation_summary", {})
        contributing = perturbation_result.get("contributing_chunks", [])
        per_chunk_results = perturbation_result.get("per_chunk_results", [])

        result = {
            # Identifiers
            "question": question,
            "ground_truth": ground_truth,
            **(metadata or {}),
            # Pipeline outputs
            "generated_answer": answer,
            "num_chunks_retrieved": len(search_results),
            "top_retrieval_score": max(scores) if scores else 0.0,
            "avg_retrieval_score": sum(scores) / len(scores) if scores else 0.0,
            # ── Top 5 retrieved chunks ──
            "retrieved_chunks": [
                {
                    "rank": i + 1,
                    "score": round(r.get("score", 0.0), 4),
                    "page": r.get("page_number", -1),
                    "paragraph": r.get("paragraph_number", -1),
                    "text_preview": r.get("text", "")[:200].strip(),
                }
                for i, r in enumerate(search_results[:5])
            ],
            **{
                f"chunk{i+1}_score": round(search_results[i].get("score", 0.0), 4)
                if i < len(search_results) else 0.0
                for i in range(5)
            },
            **{
                f"chunk{i+1}_text": search_results[i].get("text", "")[:150].strip()
                if i < len(search_results) else ""
                for i in range(5)
            },
            # ── Retrieval accuracy ──
            "hit_rate": hr,
            "mrr": mrr,
            "precision_at_k": p_at_k,
            # ── Generation accuracy ──
            "exact_match": em,
            "f1": f1,
            "rouge_l": rl,
            "answer_contains": contains,
            # ── Layer 1: Source Attribution ──
            "l1_attribution_precision": round(sa.precision, 4),
            "l1_attribution_recall": round(sa.recall, 4),
            "l1_attribution_f1": round(sa.f1_score, 4),
            # ── Layer 1: Provenance Completeness ──
            "l1_provenance_completeness": round(pc.completeness_score, 4),
            "l1_provenance_granularity": round(pc.granularity_score, 4),
            # ── Layer 1: Confidence Calibration ──
            "l1_confidence_score": round(cc.confidence_score, 4),
            "l1_confidence_tier": cc.confidence_tier,
            # ── Layer 1: Explanation Faithfulness ──
            "l1_faithfulness": round(ef.overall_faithfulness, 4),
            "l1_grounding_ratio": round(ef.answer_grounding_ratio, 4),
            # ── LIME on retrieval ──
            "lime_query_sensitivity": lime_result.get("query_sensitivity", 0.0),
            "lime_stability_score": lime_result.get("stability_score", 0.0),
            "lime_most_important_words": lime_result.get("most_important_words", []),
            "lime_least_important_words": lime_result.get("least_important_words", []),
            "lime_word_importance": lime_result.get("word_importance", []),
            "lime_total_perturbations": lime_result.get("total_perturbations", 0),
            # ── Perturbation testing on generation ──
            "perturbation_chunks_tested": p_summary.get("total_chunks_tested", 0),
            "perturbation_chunks_contributed": p_summary.get("chunks_that_contributed", 0),
            "perturbation_chunks_not_used": p_summary.get("chunks_not_used", 0),
            "perturbation_avg_change_ratio": p_summary.get("avg_change_ratio", 0.0),
            "perturbation_top_contributing_rank": p_summary.get("top_contributing_chunk_rank"),
            # Top contributing chunks (structured — in JSON output)
            "contributing_chunks": contributing,
            # Per-chunk change ratios as flat columns for Excel
            **{
                f"chunk{i+1}_contributed": per_chunk_results[i]["contributed"]
                if i < len(per_chunk_results) else False
                for i in range(5)
            },
            **{
                f"chunk{i+1}_change_ratio": per_chunk_results[i]["change_ratio"]
                if i < len(per_chunk_results) else 0.0
                for i in range(5)
            },
            # ── LLM-as-judge (optional) ──
            "judge_faithfulness": judge["faithfulness"],
            "judge_relevancy": judge["relevancy"],
            "judge_correctness": judge["correctness"],
            # ── Timing ──
            "end_to_end_ms": round(elapsed_ms, 2),
        }

        return result

    # ============================================================
    # BATCH EVALUATION
    # ============================================================

    def run_experiment(
        self,
        questions: list[dict],
        document_id: str,
        experiment_name: str = "experiment",
        top_k: int = 5,
    ) -> dict:
        logger.info(f"=== Running {experiment_name}: {len(questions)} questions ===")
        results = []

        for i, q in enumerate(questions):
            question = q.get("question", q.get("statement", ""))
            ground_truth = q.get("ground_truth", [q.get("answer", "")])
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]

            metadata = {
                k: v for k, v in q.items()
                if k not in ("question", "ground_truth", "answer")
            }

            try:
                result = self.evaluate_single_query(
                    question=question,
                    ground_truth=ground_truth,
                    document_id=document_id,
                    top_k=top_k,
                    metadata=metadata,
                )
                results.append(result)

                # Per-query LIME + perturbation console log
                top_words = result.get("lime_most_important_words", [])
                top_word_str = ", ".join(
                    f"{w['word']}({w['importance']:.2f})"
                    for w in top_words[:3]
                ) if top_words else "N/A"

                contrib = result.get("contributing_chunks", [])
                contrib_str = ", ".join(
                    f"Chunk{c['rank']}(Δ{c['change_ratio']:.2f})"
                    for c in contrib[:3]
                ) if contrib else "none"

                logger.info(
                    f"  Q{i+1}: LIME→[{top_word_str}] | "
                    f"Perturb→contributing:[{contrib_str}] | "
                    f"EM={result.get('exact_match',0):.0f} F1={result.get('f1',0):.2f}"
                )

                if (i + 1) % 10 == 0 or (i + 1) == len(questions):
                    partial_agg = compute_aggregate_metrics(results)
                    logger.info(
                        f"  [{i+1}/{len(questions)}] "
                        f"EM={partial_agg.get('exact_match',{}).get('mean',0):.3f}, "
                        f"F1={partial_agg.get('f1',{}).get('mean',0):.3f}, "
                        f"HR={partial_agg.get('hit_rate',{}).get('mean',0):.3f}, "
                        f"LIME_sens={partial_agg.get('lime_query_sensitivity',{}).get('mean',0):.3f}, "
                        f"P_contrib={partial_agg.get('perturbation_chunks_contributed',{}).get('mean',0):.2f}"
                    )

            except Exception as e:
                logger.error(f"  Error on question {i+1}: {e}")
                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "error": str(e),
                    **{k: 0.0 for k in [
                        "hit_rate", "mrr", "precision_at_k",
                        "exact_match", "f1", "rouge_l", "answer_contains",
                        "l1_attribution_precision", "l1_attribution_recall", "l1_attribution_f1",
                        "l1_provenance_completeness", "l1_provenance_granularity",
                        "l1_confidence_score", "l1_faithfulness", "l1_grounding_ratio",
                        "lime_query_sensitivity", "lime_stability_score",
                        "perturbation_chunks_contributed", "perturbation_avg_change_ratio",
                        "judge_faithfulness", "judge_relevancy", "judge_correctness",
                    ]},
                })

        aggregated = compute_aggregate_metrics(results)

        output = {
            "experiment": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
            "successful": len([r for r in results if "error" not in r]),
            "aggregated_metrics": aggregated,
            "per_query_results": results,
        }

        filename = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")

        self._save_csv_summary(results, experiment_name)
        self._print_summary(experiment_name, aggregated)

        return output

    # ============================================================
    # CSV + CONSOLE OUTPUT
    # ============================================================

    def _save_csv_summary(self, results: list[dict], experiment_name: str):
        csv_path = os.path.join(
            self.output_dir,
            f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if not results:
            return

        fieldnames = [
            # Identifiers
            "question", "generated_answer", "ground_truth", "dataset", "type",
            # Top 5 chunks
            "chunk1_score", "chunk1_text",
            "chunk2_score", "chunk2_text",
            "chunk3_score", "chunk3_text",
            "chunk4_score", "chunk4_text",
            "chunk5_score", "chunk5_text",
            # Retrieval accuracy
            "hit_rate", "mrr", "precision_at_k",
            # Generation accuracy
            "exact_match", "f1", "rouge_l", "answer_contains",
            # Layer 1: Source Attribution
            "l1_attribution_precision", "l1_attribution_recall", "l1_attribution_f1",
            # Layer 1: Provenance
            "l1_provenance_completeness", "l1_provenance_granularity",
            # Layer 1: Confidence Calibration
            "l1_confidence_score", "l1_confidence_tier",
            # Layer 1: Explanation Faithfulness
            "l1_faithfulness", "l1_grounding_ratio",
            # LIME on retrieval
            "lime_query_sensitivity", "lime_stability_score",
            "lime_most_important_words", "lime_total_perturbations",
            # Perturbation testing on generation
            "perturbation_chunks_tested",
            "perturbation_chunks_contributed",
            "perturbation_chunks_not_used",
            "perturbation_avg_change_ratio",
            "perturbation_top_contributing_rank",
            # Per-chunk contribution (flat)
            "chunk1_contributed", "chunk1_change_ratio",
            "chunk2_contributed", "chunk2_change_ratio",
            "chunk3_contributed", "chunk3_change_ratio",
            "chunk4_contributed", "chunk4_change_ratio",
            "chunk5_contributed", "chunk5_change_ratio",
            # LLM-as-judge
            "judge_faithfulness", "judge_relevancy", "judge_correctness",
            # Meta
            "end_to_end_ms",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = {**r}
                row["ground_truth"] = "|".join(row.get("ground_truth", []))
                writer.writerow(row)

        logger.info(f"CSV summary saved to {csv_path}")

    def _print_summary(self, experiment_name: str, aggregated: dict):
        logger.info(f"\n{'='*70}")
        logger.info(f"  RESULTS: {experiment_name}")
        logger.info(f"{'='*70}")

        groups = {
            "── Retrieval Accuracy ──": ["hit_rate", "mrr", "precision_at_k"],
            "── Generation Accuracy ──": ["exact_match", "f1", "rouge_l"],
            "── Layer 1: Source Attribution ──": [
                "l1_attribution_precision", "l1_attribution_recall", "l1_attribution_f1"
            ],
            "── Layer 1: Provenance ──": [
                "l1_provenance_completeness", "l1_provenance_granularity"
            ],
            "── Layer 1: Confidence Calibration ──": ["l1_confidence_score"],
            "── Layer 1: Explanation Faithfulness ──": [
                "l1_faithfulness", "l1_grounding_ratio"
            ],
            "── LIME: Retrieval Word Importance ──": [
                "lime_query_sensitivity", "lime_stability_score",
            ],
            "── Perturbation: Chunk Contribution ──": [
                "perturbation_chunks_contributed",
                "perturbation_chunks_not_used",
                "perturbation_avg_change_ratio",
            ],
        }
        if self.use_llm_judge:
            groups["── LLM-as-Judge ──"] = [
                "judge_faithfulness", "judge_relevancy", "judge_correctness"
            ]

        logger.info(f"{'Metric':<40} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        logger.info(f"{'-'*70}")
        for group_label, keys in groups.items():
            logger.info(f"  {group_label}")
            for key in keys:
                if key in aggregated:
                    v = aggregated[key]
                    logger.info(
                        f"  {key:<38} {v['mean']:>8.4f} {v['std']:>8.4f} "
                        f"{v['min']:>8.4f} {v['max']:>8.4f}"
                    )
        logger.info(f"{'='*70}\n")

    def cleanup_collection(self):
        self._cleanup_eval_collection(self.vector_store.collection_prefix)