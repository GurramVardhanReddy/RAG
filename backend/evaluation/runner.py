"""
Evaluation Runner for Explainable RAG Pipeline
=================================================
Orchestrates end-to-end evaluation on standard benchmarks
and custom multi-type questions.

Updated for v0.3.0 domain-aware VectorStoreService:
    - Uses collection_prefix (not collection_name)
    - Passes sparse_vectors + domain to add_documents()
    - Uses domain-aware search()

Usage:
    python run_evaluation.py --experiment benchmarks --n 50 --no-judge

This module directly reuses your existing pipeline components:
    - VectorStoreService (Qdrant, domain-aware collections)
    - get_embedding_model (BAAI/bge-base-en-v1.5)
    - generate_answer (GPT-4o)
    - ExplainabilityEngine
"""

import os
import json
import time
import csv
from datetime import datetime
from typing import Optional
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
    ):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.output_dir = output_dir
        self.use_llm_judge = use_llm_judge
        self.eval_domain = "General"  # Benchmarks use General domain
        os.makedirs(output_dir, exist_ok=True)

        # Initialize pipeline components (reusing YOUR existing code)
        logger.info("Initializing evaluation pipeline...")
        self.embedding_model = get_embedding_model(settings.embedding_model)

        # Use collection_prefix — VectorStoreService builds domain-suffixed names
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
    # COLLECTION MANAGEMENT (for eval temp collections)
    # ============================================================

    def _set_eval_prefix(self, prefix: str):
        """Switch the vector store to use a different collection prefix for eval."""
        self.vector_store.collection_prefix = prefix
        self.vector_store._created_collections = set()  # Reset cache

    def _cleanup_eval_collection(self, prefix: str):
        """Delete the eval collection for General domain."""
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
        """
        Index a list of text passages into Qdrant for evaluation.
        Uses dense-only storage (no SPLADE for eval benchmarks to keep it simple).
        """
        from langchain_core.documents import Document
        import numpy as np

        # Ensure domain collection exists
        self.vector_store.ensure_collection_for_domain(self.eval_domain)

        # Build Document objects
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            meta.setdefault("page", i // 10)
            meta.setdefault("source", f"passage_{i}")
            docs.append(Document(page_content=text, metadata=meta))

        # Embed
        logger.info(f"Embedding {len(texts)} passages...")
        vectors = self.embedding_model.embed_documents(texts)

        # Create dummy sparse vectors (empty numpy arrays — compatible with .tolist())
        class _EmptySparse:
            indices = np.array([], dtype=np.int64)
            values = np.array([], dtype=np.float32)

        dummy_sparse = [_EmptySparse() for _ in range(len(texts))]

        # Store using the domain-aware method
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
        """
        Index a PDF/DOCX using your existing document loader + chunker.
        For Experiment 2 (custom corpus evaluation).
        """
        from backend.app.services.document_loader import load_document
        from backend.rag.chunking.chunker import semantic_chunk
        import numpy as np

        logger.info(f"Loading and indexing: {file_path}")
        docs = load_document(file_path)

        # Build page_paragraphs map
        page_paragraphs = {}
        for doc in docs:
            page_num = doc.metadata.get("page", 0)
            paragraphs = doc.metadata.get("paragraphs", [])
            page_paragraphs[page_num] = paragraphs

        # Chunk
        chunks = semantic_chunk(docs, embedding_model=self.embedding_model)
        chunk_texts = [c.page_content for c in chunks]
        vectors = self.embedding_model.embed_documents(chunk_texts)

        # Dummy sparse vectors for eval
        class _EmptySparse:
            indices = np.array([], dtype=np.int64)
            values = np.array([], dtype=np.float32)

        dummy_sparse = [_EmptySparse() for _ in range(len(chunk_texts))]

        # Ensure collection and store
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

        Args:
            prompt_mode: "domain" for UCITS, "general" for benchmarks,
                        "verification" for FEVER-style fact checking.
                        Auto-detected from metadata["type"] if not set explicitly.

        Returns dict with:
            - Pipeline outputs: answer, retrieved chunks, confidence
            - Retrieval metrics: hit_rate, mrr, precision_at_k
            - Generation metrics: exact_match, f1, rouge_l
            - LLM-judge metrics: faithfulness, relevancy, correctness
            - Timing: end_to_end_ms
        """
        start_time = time.perf_counter()

        # Auto-detect prompt mode from question type if using default
        if metadata and prompt_mode == "domain":
            q_type = metadata.get("type", "")
            q_dataset = metadata.get("dataset", "")
            if q_dataset == "fever" or q_type == "fact_verification":
                prompt_mode = "verification"
            elif q_dataset in ("nq", "hotpotqa", "triviaqa", "hotpotqa_distractor",
                               "squad", "boolq", "coqa"):
                prompt_mode = "general"
            elif q_dataset == "custom":
                prompt_mode = "general"

        # 1. Embed query
        query_vector = self.embedding_model.embed_query(question)

        # 2. Retrieve (dense-only search, using domain-aware API)
        search_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            document_id=document_id,
            domain=self.eval_domain,
        )

        retrieved_texts = [r["text"] for r in search_results]
        scores = [r["score"] for r in search_results]

        # 3. Generate answer (with the appropriate prompt mode)
        answer = generate_answer(
            question=question,
            search_results=search_results,
            model_name="gpt-4o",
            temperature=0,
            api_key=self.api_key,
            prompt_mode=prompt_mode,
        )

        # 4. Explainability
        confidence = self.explainability_engine.compute_confidence_score(
            scores, answer, question
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # 5. Compute retrieval metrics
        hr = hit_rate(retrieved_texts, ground_truth)
        mrr = mean_reciprocal_rank(retrieved_texts, ground_truth)
        p_at_k = precision_at_k(retrieved_texts, ground_truth, k=top_k)

        # 6. Normalize answer for structured-output benchmarks before scoring
        #    BoolQ: GPT-4o says "Yes"/"No", ground truth is "True"/"False"
        #    FEVER: GPT-4o gives a sentence, ground truth is SUPPORTS/REFUTES/NOT ENOUGH INFO
        scored_answer = answer
        q_dataset = (metadata or {}).get("dataset", "")
        q_type = (metadata or {}).get("type", "")

        if q_dataset == "boolq" or q_type == "fact_verification" and q_dataset == "boolq":
            ans_lower = answer.lower().strip()
            if any(w in ans_lower for w in ["yes", "true", "correct", "indeed", "affirmative"]):
                scored_answer = "True"
            elif any(w in ans_lower for w in ["no", "false", "incorrect", "not true", "untrue"]):
                scored_answer = "False"

        elif q_dataset == "fever":
            ans_lower = answer.lower().strip()
            if any(w in ans_lower for w in ["supports", "supported", "true", "correct", "confirms", "confirmed"]):
                scored_answer = "SUPPORTS"
            elif any(w in ans_lower for w in ["refutes", "refuted", "false", "incorrect", "contradicts", "contradicted"]):
                scored_answer = "REFUTES"
            elif any(w in ans_lower for w in ["not enough", "insufficient", "cannot determine", "can't determine", "nei"]):
                scored_answer = "NOT ENOUGH INFO"

        # 7. Compute generation metrics
        em = exact_match(scored_answer, ground_truth)
        f1 = token_f1(scored_answer, ground_truth)
        rl = rouge_l(scored_answer, ground_truth)
        contains = answer_contains_ground_truth(scored_answer, ground_truth)

        # 7. LLM-as-judge (optional — costs API calls)
        judge = {"faithfulness": 0.0, "relevancy": 0.0, "correctness": 0.0}
        if self.use_llm_judge:
            judge = llm_judge_scores(
                question=question,
                answer=answer,
                contexts=retrieved_texts,
                ground_truth=ground_truth[0] if ground_truth else "",
                api_key=self.api_key,
            )

        result = {
            # Identifiers
            "question": question,
            "ground_truth": ground_truth,
            **(metadata or {}),
            # Pipeline outputs
            "generated_answer": answer,
            "confidence_score": confidence,
            "num_chunks_retrieved": len(search_results),
            "top_retrieval_score": max(scores) if scores else 0.0,
            "avg_retrieval_score": sum(scores) / len(scores) if scores else 0.0,
            # Retrieval metrics
            "hit_rate": hr,
            "mrr": mrr,
            "precision_at_k": p_at_k,
            # Generation metrics
            "exact_match": em,
            "f1": f1,
            "rouge_l": rl,
            "answer_contains": contains,
            # LLM-judge metrics
            "faithfulness": judge["faithfulness"],
            "relevancy": judge["relevancy"],
            "correctness": judge["correctness"],
            # Timing
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
        """
        Run evaluation on a batch of questions.

        Args:
            questions: List of {"question": str, "ground_truth": list[str], ...}
            document_id: Which document to search in Qdrant
            experiment_name: Label for this experiment
            top_k: Number of chunks to retrieve

        Returns:
            Dict with per-query results and aggregated metrics
        """
        logger.info(f"=== Running {experiment_name}: {len(questions)} questions ===")
        results = []

        for i, q in enumerate(questions):
            question = q.get("question", q.get("statement", ""))
            ground_truth = q.get("ground_truth", [q.get("answer", "")])
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]

            # Pass through metadata
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

                # Progress logging
                if (i + 1) % 10 == 0 or (i + 1) == len(questions):
                    partial_agg = compute_aggregate_metrics(results)
                    em_mean = partial_agg.get("exact_match", {}).get("mean", 0)
                    f1_mean = partial_agg.get("f1", {}).get("mean", 0)
                    hr_mean = partial_agg.get("hit_rate", {}).get("mean", 0)
                    logger.info(
                        f"  [{i+1}/{len(questions)}] "
                        f"EM={em_mean:.3f}, F1={f1_mean:.3f}, HR={hr_mean:.3f}"
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
                        "faithfulness", "relevancy", "correctness",
                    ]},
                })

        # Aggregate
        aggregated = compute_aggregate_metrics(results)

        output = {
            "experiment": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
            "successful": len([r for r in results if "error" not in r]),
            "aggregated_metrics": aggregated,
            "per_query_results": results,
        }

        # Save results
        filename = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")

        # Also save a CSV summary
        self._save_csv_summary(results, experiment_name)

        # Print summary
        self._print_summary(experiment_name, aggregated)

        return output

    def _save_csv_summary(self, results: list[dict], experiment_name: str):
        """Save a CSV with one row per query for easy analysis."""
        csv_path = os.path.join(
            self.output_dir,
            f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if not results:
            return

        fieldnames = [
            "question", "generated_answer", "ground_truth",
            "dataset", "type",
            "hit_rate", "mrr", "precision_at_k",
            "exact_match", "f1", "rouge_l", "answer_contains",
            "faithfulness", "relevancy", "correctness",
            "confidence_score", "end_to_end_ms",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = {**r}
                # Flatten ground_truth list for CSV
                row["ground_truth"] = "|".join(row.get("ground_truth", []))
                writer.writerow(row)

        logger.info(f"CSV summary saved to {csv_path}")

    def _print_summary(self, experiment_name: str, aggregated: dict):
        """Print a formatted summary table to the console."""
        logger.info(f"\n{'='*60}")
        logger.info(f"  RESULTS: {experiment_name}")
        logger.info(f"{'='*60}")
        logger.info(f"{'Metric':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        logger.info(f"{'-'*60}")
        for key, vals in aggregated.items():
            logger.info(
                f"{key:<22} {vals['mean']:>8.4f} {vals['std']:>8.4f} "
                f"{vals['min']:>8.4f} {vals['max']:>8.4f}"
            )
        logger.info(f"{'='*60}\n")

    # ============================================================
    # CLEANUP
    # ============================================================

    def cleanup_collection(self):
        """Delete the evaluation collection from Qdrant."""
        self._cleanup_eval_collection(self.vector_store.collection_prefix)