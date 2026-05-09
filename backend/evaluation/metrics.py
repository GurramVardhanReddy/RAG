"""
Evaluation Metrics for RAG Pipeline
=====================================
Computes retrieval and generation quality metrics.

Retrieval: Hit Rate, MRR, Precision@k, Recall@k, NDCG@k
Generation: Exact Match, F1, ROUGE-L, Faithfulness, Relevancy

References:
    - Salemi & Zamani, "Evaluating Retrieval Quality in RAG" (SIGIR 2024)
    - Joshi et al., "Robust Multi-Model RAG Pipeline" (ICAAIC 2024)
    - RAGAS framework (Es et al., 2023)
"""

import re
import json
import string
import numpy as np
from typing import Optional
from collections import Counter
from loguru import logger


# ============================================================
# TEXT NORMALIZATION (standard for QA evaluation)
# ============================================================

def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    Standard normalization used in SQuAD, NQ, TriviaQA evaluations.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()


# ============================================================
# GENERATION METRICS
# ============================================================

def exact_match(prediction: str, ground_truths: list[str]) -> float:
    """
    Exact Match (EM): Does the prediction exactly match any ground truth?
    Used for: NQ, TriviaQA, HotpotQA in KILT benchmark.
    
    Returns: 1.0 if match, 0.0 otherwise
    """
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == norm_pred:
            return 1.0
    return 0.0


def token_f1(prediction: str, ground_truths: list[str]) -> float:
    """
    Token-level F1 score between prediction and best-matching ground truth.
    Standard QA metric alongside EM.
    
    Returns: F1 score (0.0 to 1.0)
    """
    norm_pred = normalize_answer(prediction)
    pred_tokens = norm_pred.split()

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens) if pred_tokens else 0
        recall = num_common / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        best_f1 = max(best_f1, f1)

    return best_f1


def rouge_l(prediction: str, ground_truths: list[str]) -> float:
    """
    ROUGE-L: Longest Common Subsequence based metric.
    Used for: Wizard of Wikipedia in KILT, long-form answers.
    
    Returns: ROUGE-L F1 (0.0 to 1.0)
    """
    def lcs_length(x: list[str], y: list[str]) -> int:
        m, n = len(x), len(y)
        table = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])
        return table[m][n]

    norm_pred = normalize_answer(prediction).split()
    best_score = 0.0

    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens or not norm_pred:
            continue
        lcs = lcs_length(norm_pred, gt_tokens)
        prec = lcs / len(norm_pred) if norm_pred else 0
        rec = lcs / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        best_score = max(best_score, f1)

    return best_score


def answer_contains_ground_truth(prediction: str, ground_truths: list[str]) -> float:
    """
    Soft match: Does the prediction contain any ground truth answer?
    More forgiving than EM — useful for longer generated answers.
    
    Returns: 1.0 if contained, 0.0 otherwise
    """
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) in norm_pred:
            return 1.0
    return 0.0


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def hit_rate(retrieved_texts: list[str], ground_truth_answers: list[str]) -> float:
    """
    Hit Rate (HR): Is the answer present in ANY retrieved chunk?
    
    From Joshi et al. (ICAAIC 2024): Binary score indicating whether
    the right chunk is present in the retrieved chunks.
    
    Returns: 1.0 if hit, 0.0 otherwise
    """
    for chunk_text in retrieved_texts:
        norm_chunk = normalize_answer(chunk_text)
        for gt in ground_truth_answers:
            if normalize_answer(gt) in norm_chunk:
                return 1.0
    return 0.0


def mean_reciprocal_rank(retrieved_texts: list[str], ground_truth_answers: list[str]) -> float:
    """
    Mean Reciprocal Rank (MRR): Reciprocal of the rank of the first relevant chunk.
    
    From Joshi et al. (ICAAIC 2024) and Salemi & Zamani (SIGIR 2024).
    MRR = 1/rank of first relevant result.
    
    Returns: 1/rank (0.0 if no relevant chunk found)
    """
    for rank, chunk_text in enumerate(retrieved_texts, 1):
        norm_chunk = normalize_answer(chunk_text)
        for gt in ground_truth_answers:
            if normalize_answer(gt) in norm_chunk:
                return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_texts: list[str], ground_truth_answers: list[str], k: int = 5) -> float:
    """
    Precision@k: Fraction of top-k retrieved chunks that are relevant.
    From Salemi & Zamani (SIGIR 2024).
    """
    relevant = 0
    for chunk_text in retrieved_texts[:k]:
        norm_chunk = normalize_answer(chunk_text)
        for gt in ground_truth_answers:
            if normalize_answer(gt) in norm_chunk:
                relevant += 1
                break
    return relevant / min(k, len(retrieved_texts)) if retrieved_texts else 0.0


# ============================================================
# LLM-AS-JUDGE METRICS (Following Joshi et al.)
# ============================================================

def llm_judge_scores(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    api_key: str,
    model_name: str = "gpt-4o",
) -> dict[str, float]:
    """
    LLM-as-a-judge evaluation for faithfulness, relevancy, correctness.
    
    Following Joshi et al. (ICAAIC 2024) who used GPT-4 for response evaluation.
    
    Returns: {"faithfulness": float, "relevancy": float, "correctness": float}
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
    context_str = "\n---\n".join(contexts[:3])  # Limit context for cost

    prompt = f"""You are evaluating a RAG system's response. Score each dimension from 0.0 to 1.0.
Return ONLY valid JSON (no markdown, no backticks):
{{"faithfulness": <float>, "relevancy": <float>, "correctness": <float>}}

Definitions:
- faithfulness: Is the answer supported by the provided context? (1.0 = fully supported)
- relevancy: Does the answer address the question? (1.0 = directly answers it)
- correctness: Is the answer factually correct compared to the ground truth? (1.0 = fully correct)

Question: {question}
Context provided: {context_str[:2000]}
Generated answer: {answer}
Ground truth: {ground_truth}

JSON scores:"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        scores = json.loads(content)
        return {
            "faithfulness": float(scores.get("faithfulness", 0.0)),
            "relevancy": float(scores.get("relevancy", 0.0)),
            "correctness": float(scores.get("correctness", 0.0)),
        }
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return {"faithfulness": 0.0, "relevancy": 0.0, "correctness": 0.0}


# ============================================================
# AGGREGATE METRICS
# ============================================================

def compute_aggregate_metrics(results: list[dict]) -> dict:
    """
    Compute aggregate metrics across all evaluation results.
    
    Args:
        results: List of per-query result dicts from runner.py
    
    Returns:
        Dict of aggregated metrics with means and standard deviations
    """
    if not results:
        return {}

    metric_keys = [
        "exact_match", "f1", "rouge_l", "answer_contains",
        "hit_rate", "mrr", "precision_at_k",
        "faithfulness", "relevancy", "correctness",
    ]

    aggregated = {}
    for key in metric_keys:
        values = [r.get(key, 0.0) for r in results if key in r]
        if values:
            aggregated[key] = {
                "mean": round(np.mean(values), 4),
                "std": round(np.std(values), 4),
                "min": round(np.min(values), 4),
                "max": round(np.max(values), 4),
                "count": len(values),
            }

    return aggregated
