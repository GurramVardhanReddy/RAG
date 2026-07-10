"""
Dataset Loaders for RAG Evaluation
====================================
Uses SELF-CONTAINED benchmarks where each question comes bundled
with its own context passages. No need to index 22M Wikipedia passages.

Datasets:
    - HotpotQA (distractor): 10 paragraphs per question (2 gold + 8 distractors)
    - SQuAD 2.0: Each question comes with its Wikipedia paragraph
    - BoolQ: Yes/No fact verification with passage context
    - FEVER (standalone): Fact verification with evidence sentences
    - CoQA: Conversational QA across 7 domains with passage context
    - Custom multi-type: GPT-4o generated questions from your own corpus

References:
    - Yang et al., "HotpotQA" (EMNLP 2018)
    - Rajpurkar et al., "SQuAD 2.0" (ACL 2018)
    - Clark et al., "BoolQ" (NAACL 2019)
    - Thorne et al., "FEVER: Fact Extraction and VERification" (NAACL 2018)
    - Reddy et al., "CoQA: Conversational Question Answering" (TACL 2019)
    - Joshi et al., "Robust Multi-Model RAG Pipeline" (ICAAIC 2024)
    - Salemi and Zamani, "Evaluating Retrieval Quality in RAG" (SIGIR 2024)
"""

import json
import os
import random
from typing import Optional
from loguru import logger


# ============================================================
# 1. HotpotQA (distractor) — Multi-hop QA
# ============================================================

def load_hotpotqa_distractor(n: int = 200, split: str = "validation", offset: int = 0) -> dict:
    """
    Load HotpotQA distractor setting.
    Each question comes with 10 paragraphs (2 gold + 8 distractors).
    Tests multi-hop reasoning and retrieval across multiple passages.

    Args:
        offset: Skip this many questions before starting (for batched runs).

    Reference: Yang et al., "HotpotQA" (EMNLP 2018)
    """
    from datasets import load_dataset
    logger.info(f"Loading HotpotQA distractor ({split}, n={n})...")

    # Try multiple sources (original host went offline May 2025)
    ds = None
    for dataset_id in ["hotpotqa/hotpot_qa", "vincentkoc/hotpot_qa_archive"]:
        try:
            ds = load_dataset(dataset_id, "distractor", split=split, streaming=True)
            logger.info(f"Loaded HotpotQA from '{dataset_id}'")
            break
        except Exception as e:
            logger.warning(f"HotpotQA load from '{dataset_id}' failed: {e}")

    if ds is None:
        logger.error("Could not load HotpotQA from any source")
        return {"questions": [], "corpus": []}

    questions = []
    corpus = []
    corpus_seen = set()
    skipped = 0
    for ex in ds:
        if skipped < offset:
            skipped += 1
            continue
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
            "question": ex["question"],
            "ground_truth": [ex["answer"]],
            "gold_titles": list(gold_titles),
            "supporting_facts": {
                "titles": ex["supporting_facts"]["title"],
                "sent_ids": ex["supporting_facts"]["sent_id"],
            },
            "dataset": "hotpotqa_distractor",
            "type": "multi_hop",
        })
    logger.info(f"Loaded {len(questions)} questions with {len(corpus)} passages ({sum(1 for c in corpus if c['is_gold'])} gold)")
    return {"questions": questions, "corpus": corpus}


# ============================================================
# 2. SQuAD 2.0 — Reading Comprehension + Unanswerable
# ============================================================

def load_squad(n: int = 200, split: str = "validation", offset: int = 0) -> dict:
    """
    Load SQuAD 2.0 — each question comes with its Wikipedia paragraph.
    Includes unanswerable questions (ground_truth = ["unanswerable"]).

    Args:
        offset: Skip this many questions before starting (for batched runs).

    Reference: Rajpurkar et al., "SQuAD 2.0" (ACL 2018)
    """
    from datasets import load_dataset
    logger.info(f"Loading SQuAD 2.0 ({split}, n={n}, offset={offset})...")
    ds = load_dataset("rajpurkar/squad_v2", split=split, streaming=True)
    questions = []
    corpus = []
    corpus_seen = set()
    skipped = 0
    for ex in ds:
        if skipped < offset:
            skipped += 1
            continue
        if len(questions) >= n:
            break
        ctx_key = ex["context"][:100]
        if ctx_key not in corpus_seen:
            corpus_seen.add(ctx_key)
            corpus.append({"text": ex["context"], "title": ex["title"], "is_gold": True})
        answers = ex["answers"]["text"]
        if answers:
            questions.append({
                "question": ex["question"],
                "ground_truth": list(set(answers)),
                "context_title": ex["title"],
                "dataset": "squad",
                "type": "short_form",
            })
        else:
            questions.append({
                "question": ex["question"],
                "ground_truth": ["unanswerable"],
                "context_title": ex["title"],
                "dataset": "squad",
                "type": "unanswerable",
            })
    logger.info(f"Loaded {len(questions)} questions with {len(corpus)} passages")
    return {"questions": questions, "corpus": corpus}


# ============================================================
# 3. BoolQ — Yes/No Fact Verification
# ============================================================

def load_boolq(n: int = 200, split: str = "validation", offset: int = 0) -> dict:
    """
    Load BoolQ — yes/no questions with passage context.
    Tests binary reasoning / fact verification ability.

    Args:
        offset: Skip this many questions before starting (for batched runs).

    Reference: Clark et al., "BoolQ" (NAACL 2019)
    """
    from datasets import load_dataset
    logger.info(f"Loading BoolQ ({split}, n={n}, offset={offset})...")
    ds = load_dataset("google/boolq", split=split, streaming=True)
    questions = []
    corpus = []
    corpus_seen = set()
    skipped = 0
    for ex in ds:
        if skipped < offset:
            skipped += 1
            continue
        if len(questions) >= n:
            break
        passage = ex["passage"]
        passage_key = passage[:100]
        if passage_key not in corpus_seen:
            corpus_seen.add(passage_key)
            corpus.append({"text": passage, "title": "", "is_gold": True})
        questions.append({
            "question": ex["question"],
            "ground_truth": ["True" if ex["answer"] else "False"],
            "dataset": "boolq",
            "type": "fact_verification",
        })
    logger.info(f"Loaded {len(questions)} questions with {len(corpus)} passages")
    return {"questions": questions, "corpus": corpus}


# ============================================================
# 4. FEVER (standalone) — Fact Verification with Evidence
# ============================================================

def load_fever(n: int = 200, split: str = "labelled_dev", offset: int = 0) -> dict:
    """
    Load FEVER — claim verification with evidence sentences.
    Each claim is labelled SUPPORTS, REFUTES, or NOT ENOUGH INFO.
    Evidence sentences are bundled with the claim.

    Self-contained: uses the evidence provided with each claim,
    no need to index the full FEVER Wikipedia dump.

    Args:
        offset: Skip this many claims before starting (for batched runs).

    Reference: Thorne et al., "FEVER" (NAACL 2018)
    HuggingFace: fever/fever (paper_dev / paper_test / labelled_dev)
    """
    from datasets import load_dataset
    logger.info(f"Loading FEVER ({split}, n={n})...")

    # Try multiple loading approaches (HF dataset configs change)
    dataset = None
    load_errors = []

    # Approach 1: Load from fever/fever with split
    try:
        dataset = load_dataset("fever/fever", "v1.0", split=split, streaming=True)
        logger.info("Loaded FEVER via fever/fever v1.0")
    except Exception as e:
        load_errors.append(f"fever/fever v1.0: {e}")

    # Approach 2: Try the wiki_pages-free version
    if dataset is None:
        try:
            dataset = load_dataset("fever/fever", split=split, streaming=True)
            logger.info("Loaded FEVER via fever/fever (default)")
        except Exception as e:
            load_errors.append(f"fever/fever default: {e}")

    # Approach 3: Try copenlu/fever_gold_evidence which has evidence built in
    if dataset is None:
        try:
            dataset = load_dataset("copenlu/fever_gold_evidence", split="validation",
                                    streaming=True)
            logger.info("Loaded FEVER via copenlu/fever_gold_evidence")
        except Exception as e:
            load_errors.append(f"copenlu/fever_gold_evidence: {e}")

    if dataset is None:
        for err in load_errors:
            logger.warning(f"  FEVER load attempt failed: {err}")
        logger.error("Could not load FEVER from any source. Returning empty dataset.")
        return {"questions": [], "corpus": []}

    questions = []
    corpus = []
    corpus_seen = set()
    valid_seen = 0  # counts only SUPPORTS/REFUTES claims (what offset is against)

    for ex in dataset:
        if len(questions) >= n:
            break

        # Get the claim and label
        claim = ex.get("claim", "")
        label = ex.get("label", "")

        # Normalize label to string
        if isinstance(label, int):
            label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
            label = label_map.get(label, "NOT ENOUGH INFO")
        elif isinstance(label, str):
            label = label.upper()

        if not claim or not label:
            continue

        # Skip NOT ENOUGH INFO — no real evidence to retrieve from
        if label == "NOT ENOUGH INFO":
            continue

        # Apply offset against valid (non-NEI) questions
        if valid_seen < offset:
            valid_seen += 1
            continue
        valid_seen += 1

        # Extract evidence text
        # copenlu/fever_gold_evidence schema: evidence = [["Wikipedia_Title", "sentence"], ...]
        # fever/fever schema: evidence_sentence field or evidence_wiki_url + sentence
        evidence_text = ""

        raw_evidence = ex.get("evidence", None)

        if raw_evidence and isinstance(raw_evidence, list):
            # copenlu format: list of ["Wikipedia_Title", "sentence_index", "sentence_text"]
            sentences = []
            for item in raw_evidence:
                if isinstance(item, list) and len(item) >= 3:
                    # item[0] = title, item[1] = sentence index, item[2] = sentence text
                    sentences.append(str(item[2]))
                elif isinstance(item, list) and len(item) == 2:
                    sentences.append(str(item[1]))
                elif isinstance(item, str):
                    sentences.append(item)
            evidence_text = " ".join(sentences).strip()

        elif "evidence_sentence" in ex and ex["evidence_sentence"]:
            evidence_text = ex["evidence_sentence"]

        elif "evidence_wiki_url" in ex:
            title = ex.get("evidence_wiki_url", "").replace("_", " ")
            sentence = ex.get("evidence_sentence", "")
            evidence_text = f"{title}: {sentence}" if sentence else title

        # Only add questions with real evidence (no placeholders)
        if not evidence_text or len(evidence_text.strip()) < 20:
            continue

        ev_key = evidence_text[:100]
        if ev_key not in corpus_seen:
            corpus_seen.add(ev_key)
            corpus.append({
                "text": evidence_text,
                "title": ex.get("evidence_wiki_url", "").replace("_", " "),
                "is_gold": True,
            })

        questions.append({
            "question": claim,
            "ground_truth": [label],
            "dataset": "fever",
            "type": "fact_verification",
        })

    logger.info(
        f"Loaded {len(questions)} FEVER claims with {len(corpus)} evidence passages "
        f"(labels: {sum(1 for q in questions if q['ground_truth'][0] == 'SUPPORTS')} SUPPORTS, "
        f"{sum(1 for q in questions if q['ground_truth'][0] == 'REFUTES')} REFUTES, "
        f"{sum(1 for q in questions if q['ground_truth'][0] == 'NOT ENOUGH INFO')} NEI)"
    )
    return {"questions": questions, "corpus": corpus}


# ============================================================
# 5. CoQA — Conversational QA (7 domains)
# ============================================================

def load_coqa(n: int = 200, offset: int = 0) -> dict:
    """
    Load CoQA — conversational QA across 7 diverse domains.
    Each story comes with passage context + multiple Q&A turns.
    We extract individual Q&A pairs with their passage.

    Domains: children's stories, literature, mid/high school exams,
    news, Wikipedia, Reddit, science.

    Self-contained: each question comes with its passage.

    Args:
        offset: Skip this many flattened Q&A pairs before starting
                (for batched runs).

    Reference: Reddy et al., "CoQA" (TACL 2019)
    """
    from datasets import load_dataset
    logger.info(f"Loading CoQA (n={n}, offset={offset})...")

    dataset = None
    load_errors = []

    # Approach 1: stanfordnlp/coqa
    try:
        dataset = load_dataset("stanfordnlp/coqa", split="validation", streaming=True)
        logger.info("Loaded CoQA via stanfordnlp/coqa")
    except Exception as e:
        load_errors.append(f"stanfordnlp/coqa: {e}")

    # Approach 2: coqa (direct)
    if dataset is None:
        try:
            dataset = load_dataset("coqa", split="validation", streaming=True)
            logger.info("Loaded CoQA via coqa")
        except Exception as e:
            load_errors.append(f"coqa: {e}")

    if dataset is None:
        for err in load_errors:
            logger.warning(f"  CoQA load attempt failed: {err}")
        logger.error("Could not load CoQA from any source. Returning empty dataset.")
        return {"questions": [], "corpus": []}

    questions = []
    corpus = []
    corpus_seen = set()
    skipped = 0

    for ex in dataset:
        if len(questions) >= n:
            break

        story = ex.get("story", "")
        source = ex.get("source", "unknown")

        if not story or len(story.strip()) < 50:
            continue

        # Add story to corpus (deduplicated)
        story_key = story[:100]
        if story_key not in corpus_seen:
            corpus_seen.add(story_key)
            corpus.append({
                "text": story,
                "title": source,
                "is_gold": True,
            })

        # Extract Q&A pairs from this story
        q_list = ex.get("questions", [])
        a_list = ex.get("answers", {})

        # CoQA answers structure: {"input_text": [...], "answer_start": [...], ...}
        answer_texts = a_list.get("input_text", []) if isinstance(a_list, dict) else []

        for i, q_text in enumerate(q_list):
            if len(questions) >= n:
                break
            if isinstance(q_text, dict):
                q_text = q_text.get("input_text", str(q_text))

            answer = answer_texts[i] if i < len(answer_texts) else ""
            if not q_text or not answer:
                continue

            # Skip the first `offset` flattened Q&A pairs
            if skipped < offset:
                skipped += 1
                continue

            questions.append({
                "question": q_text,
                "ground_truth": [answer],
                "dataset": "coqa",
                "type": "conversational",
                "source_domain": source,
            })

    # Count domain distribution
    domain_counts = {}
    for q in questions:
        d = q.get("source_domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    logger.info(
        f"Loaded {len(questions)} CoQA questions from {len(corpus)} stories. "
        f"Domains: {domain_counts}"
    )
    return {"questions": questions, "corpus": corpus}


# ============================================================
# 6. Natural Questions (open, no context — optional)
# ============================================================

def load_natural_questions(n: int = 200, split: str = "validation") -> list[dict]:
    """
    Load Natural Questions (open variant).
    NOTE: No context passages included — this is open-domain.
    Use only if you have a Wikipedia index, or for parametric-only testing.
    """
    from datasets import load_dataset
    logger.info(f"Loading Natural Questions ({split}, n={n})...")
    ds = load_dataset("google-research-datasets/nq_open", split=split)
    subset = ds.select(range(min(n, len(ds))))
    results = []
    for ex in subset:
        results.append({
            "question": ex["question"],
            "ground_truth": ex["answer"],
            "dataset": "nq",
            "type": "short_form",
        })
    logger.info(f"Loaded {len(results)} NQ questions")
    return results


# ============================================================
# 7. Custom Multi-Type Question Generation (GPT-4o)
# ============================================================

def generate_multi_type_questions(
    chunks: list[dict],
    api_key: str,
    n_per_type: int = 50,
    model_name: str = "gpt-4o",
) -> dict[str, list[dict]]:
    """
    Generate evaluation questions from your own corpus using GPT-4o.
    Produces 4 types: short_form, long_form, true_false, mcq.
    
    Following Joshi et al. (ICAAIC 2024) evaluation methodology.
    """
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


# ============================================================
# UTILITIES
# ============================================================

def save_questions(questions: dict, output_path: str):
    """Save generated questions to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)
    logger.info(f"Saved questions to {output_path}")


def load_questions(path: str) -> dict:
    """Load previously generated questions from JSON."""
    with open(path, "r") as f:
        return json.load(f)