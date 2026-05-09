"""
Dataset Loaders for RAG Evaluation
====================================
Uses SELF-CONTAINED benchmarks where each question comes bundled
with its own context passages. No need to index 22M Wikipedia passages.

Datasets:
    - HotpotQA (distractor): 10 paragraphs per question (2 gold + 8 distractors)
    - SQuAD 2.0: Each question comes with its Wikipedia paragraph
    - BoolQ: Yes/No fact verification with passage context
    - Custom multi-type: GPT-4o generated questions from your own corpus

References:
    - Yang et al., "HotpotQA" (EMNLP 2018)
    - Rajpurkar et al., "SQuAD 2.0" (ACL 2018)
    - Clark et al., "BoolQ" (NAACL 2019)
    - Joshi et al., "Robust Multi-Model RAG Pipeline" (ICAAIC 2024)
    - Salemi and Zamani, "Evaluating Retrieval Quality in RAG" (SIGIR 2024)
"""

import json
import os
import random
from typing import Optional
from loguru import logger


def load_hotpotqa_distractor(n: int = 200, split: str = "validation") -> dict:
    from datasets import load_dataset
    logger.info(f"Loading HotpotQA distractor ({split}, n={n})...")
    ds = load_dataset("hotpot_qa", "distractor", split=split, streaming=True)
    questions = []
    corpus = []
    corpus_seen = set()
    for ex in ds:
        if len(questions) >= n:
            break
        gold_titles = set(ex["supporting_facts"]["title"])
        for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
            passage_text = " ".join(sentences)
            passage_key = f"{title}::{passage_text[:100]}"
            if passage_key not in corpus_seen and len(passage_text.strip()) > 30:
                corpus_seen.add(passage_key)
                corpus.append({"text": passage_text, "title": title, "is_gold": title in gold_titles})
        questions.append({
            "question": ex["question"], "ground_truth": [ex["answer"]],
            "gold_titles": list(gold_titles),
            "supporting_facts": {"titles": ex["supporting_facts"]["title"], "sent_ids": ex["supporting_facts"]["sent_id"]},
            "dataset": "hotpotqa_distractor", "type": "multi_hop",
        })
    logger.info(f"Loaded {len(questions)} questions with {len(corpus)} passages ({sum(1 for c in corpus if c['is_gold'])} gold)")
    return {"questions": questions, "corpus": corpus}


def load_squad(n: int = 200, split: str = "validation") -> dict:
    from datasets import load_dataset
    logger.info(f"Loading SQuAD 2.0 ({split}, n={n})...")
    ds = load_dataset("rajpurkar/squad_v2", split=split, streaming=True)
    questions = []
    corpus = []
    corpus_seen = set()
    for ex in ds:
        if len(questions) >= n:
            break
        ctx_key = ex["context"][:100]
        if ctx_key not in corpus_seen:
            corpus_seen.add(ctx_key)
            corpus.append({"text": ex["context"], "title": ex["title"], "is_gold": True})
        answers = ex["answers"]["text"]
        if answers:
            questions.append({"question": ex["question"], "ground_truth": list(set(answers)), "context_title": ex["title"], "dataset": "squad", "type": "short_form"})
        else:
            questions.append({"question": ex["question"], "ground_truth": ["unanswerable"], "context_title": ex["title"], "dataset": "squad", "type": "unanswerable"})
    logger.info(f"Loaded {len(questions)} questions with {len(corpus)} passages")
    return {"questions": questions, "corpus": corpus}


def load_boolq(n: int = 200, split: str = "validation") -> dict:
    from datasets import load_dataset
    logger.info(f"Loading BoolQ ({split}, n={n})...")
    ds = load_dataset("google/boolq", split=split, streaming=True)
    questions = []
    corpus = []
    corpus_seen = set()
    for ex in ds:
        if len(questions) >= n:
            break
        passage = ex["passage"]
        passage_key = passage[:100]
        if passage_key not in corpus_seen:
            corpus_seen.add(passage_key)
            corpus.append({"text": passage, "title": "", "is_gold": True})
        questions.append({"question": ex["question"], "ground_truth": ["True" if ex["answer"] else "False"], "dataset": "boolq", "type": "fact_verification"})
    logger.info(f"Loaded {len(questions)} questions with {len(corpus)} passages")
    return {"questions": questions, "corpus": corpus}


def load_natural_questions(n: int = 200, split: str = "validation") -> list[dict]:
    from datasets import load_dataset
    logger.info(f"Loading Natural Questions ({split}, n={n})...")
    ds = load_dataset("nq_open", split=split)
    subset = ds.select(range(min(n, len(ds))))
    results = []
    for ex in subset:
        results.append({"question": ex["question"], "ground_truth": ex["answer"], "dataset": "nq", "type": "short_form"})
    logger.info(f"Loaded {len(results)} NQ questions")
    return results


def generate_multi_type_questions(chunks: list[dict], api_key: str, n_per_type: int = 50, model_name: str = "gpt-4o") -> dict[str, list[dict]]:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=model_name, temperature=0.3, api_key=api_key)
    all_questions = {"short_form": [], "long_form": [], "true_false": [], "mcq": []}
    sample_size = min(len(chunks), n_per_type * 2)
    sampled_chunks = random.sample(chunks, sample_size)
    logger.info(f"Generating questions from {sample_size} chunks...")
    for i, chunk in enumerate(sampled_chunks):
        chunk_text = chunk["text"] if isinstance(chunk, dict) else chunk
        if len(chunk_text.strip()) < 50:
            continue
        prompt = f"""Based on the following document excerpt, generate exactly 4 evaluation questions.
Return ONLY valid JSON (no markdown, no backticks) with this structure:
{{
    "short_form": {{"question": "...", "answer": "..."}},
    "long_form": {{"question": "...", "answer": "..."}},
    "true_false": {{"statement": "...", "answer": "True or False", "explanation": "..."}},
    "mcq": {{"question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct": "A/B/C/D", "answer": "..."}}
}}
Rules:
- short_form: Single factual question, 1-line answer
- long_form: Question requiring a detailed multi-sentence answer
- true_false: A factual statement that is either True or False
- mcq: A question with exactly 4 options, only one correct
Document excerpt:
{chunk_text[:1500]}"""
        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content)
            for qtype in ["short_form", "long_form", "true_false", "mcq"]:
                if qtype in parsed and len(all_questions[qtype]) < n_per_type:
                    q = parsed[qtype]
                    q["source_chunk"] = chunk_text[:200]
                    q["chunk_index"] = i
                    q["dataset"] = "custom"
                    q["type"] = qtype
                    all_questions[qtype].append(q)
            if (i + 1) % 10 == 0:
                counts = {k: len(v) for k, v in all_questions.items()}
                logger.info(f"  Generated {i+1}/{sample_size}: {counts}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"  Skipping chunk {i}: parse error - {e}")
            continue
        except Exception as e:
            logger.warning(f"  Skipping chunk {i}: {e}")
            continue
        if all(len(v) >= n_per_type for v in all_questions.values()):
            break
    counts = {k: len(v) for k, v in all_questions.items()}
    logger.info(f"Question generation complete: {counts}")
    return all_questions


def save_questions(questions: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)
    logger.info(f"Saved questions to {output_path}")


def load_questions(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)