# Explainable RAG — Complete Project Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [What Changed: v1 → v2](#2-what-changed-v1--v2)
3. [Full Pipeline Walkthrough](#3-full-pipeline-walkthrough)
4. [Key Terms Glossary](#4-key-terms-glossary)
5. [Evaluation Results & Findings](#5-evaluation-results--findings)
6. [File Structure](#6-file-structure)
7. [How to Run](#7-how-to-run)

---

## 1. Project Overview

**What this system does:**
Upload a PDF document (e.g., a UCITS fund supplement), ask questions about it, and get answers with full transparency — citing which page, paragraph, and source chunk the answer came from.

**Tech stack:**
| Component | Technology |
|---|---|
| API Framework | FastAPI |
| Vector Database | Qdrant (via Docker) |
| Dense Embedding | BAAI/bge-base-en-v1.5 (HuggingFace) |
| Sparse Embedding | SPLADE (prithivida/Splade_PP_en_v1 via fastembed) |
| Reranker | BAAI/bge-reranker-base (cross-encoder) |
| LLM | GPT-4o (OpenAI) |
| Evaluation | HotpotQA, SQuAD 2.0, BoolQ + LLM-as-judge |

---

## 2. What Changed: v1 → v2

### v1 (main branch — original baseline)

**Upload pipeline:**
```
PDF → Load → Semantic Chunk → BGE Dense Embedding → Qdrant (single vector)
```

**Query pipeline:**
```
Query → BGE Dense Embedding → Qdrant Cosine Search → GPT-4o → Answer
```

**What was missing:**
- Only dense (semantic) vectors — no keyword matching
- No reranking of retrieved chunks
- No multimodal support (tables, images ignored)
- Generic LLM prompt — no UCITS domain knowledge
- No HyDE
- Evaluation runner existed but was broken (wrong `add_documents` signature)

---

### v2 (feature/hybrid-search → merged to main)

**Upload pipeline:**
```
PDF → Load → Semantic Chunk
           → BGE Dense Embedding    ─┐
           → SPLADE Sparse Encoding  ├─→ Qdrant (named vectors: dense + sparse)
           → (optional) Multimodal   ─┘
              Extract tables/images → GPT-4o summary → embed → Qdrant
```

**Query pipeline:**
```
Query
  │
  ├─[HyDE ON]─→ GPT-4o generates hypothetical passage → embed hypothesis
  │
  ├─→ BGE Dense Embedding
  ├─→ SPLADE Sparse Encoding
  │
  ├─→ Qdrant Prefetch (dense: 40 candidates + sparse: 10 candidates)
  ├─→ Qdrant RRF Fusion (server-side)
  │
  ├─[Reranker ON]─→ Cross-encoder rescores all candidates
  │
  └─→ Top-K results → GPT-4o (UCITS-aware prompt) → Answer + Citations
```

---

### Side-by-side comparison

| Feature | v1 | v2 |
|---|---|---|
| Vector type | Single dense vector | Named: dense + sparse |
| Search method | Cosine similarity | Hybrid RRF (dense + sparse) |
| Reranking | None | Cross-encoder (BGE-reranker-base) |
| Query encoding | Raw query embedded | Optional HyDE: hypothesis embedded |
| Multimodal | Text only | Tables + images via GPT-4o vision |
| LLM prompt | Generic | UCITS-domain-aware |
| Qdrant collection schema | `vectors_config` (single) | `vectors_config` + `sparse_vectors_config` |
| Config settings | 7 settings | 14 settings (hybrid, reranker, HyDE, prefetch) |
| Evaluation | Broken | Fixed + mode comparison (dense/hybrid/hyde) |
| API version | v0.1.0 | v0.2.0 |

---

## 3. Full Pipeline Walkthrough

### Step A: Document Upload

**1. File validation**
Checks that the file is `.pdf`, `.docx`, or `.txt`.

**2. Document loading**
`load_document()` reads the file and extracts:
- Text content per page
- Paragraph boundaries (detected by spacing/indentation)
- Metadata: page number, source path

**3. Semantic chunking**
`semantic_chunk()` splits the document into chunks using `SemanticChunker`:
- Splits text into sentences
- Embeds each sentence using BGE
- Compares adjacent sentence embeddings
- Splits where semantic similarity drops sharply (at the 70th percentile of dissimilarity)
- Result: chunks where all sentences are about the same topic

*Why not fixed-size chunking?* Fixed-size chunks split mid-sentence or mid-concept. Semantic chunking keeps related ideas together, improving retrieval quality.

**4. Dense embedding (BGE)**
`BAAI/bge-base-en-v1.5` converts each chunk into a 768-dimensional vector.
- Captures semantic meaning
- "Shares listed on exchanges" → vector close to "public sale of units"
- Normalized for cosine similarity

**5. Sparse embedding (SPLADE)**
`prithivida/Splade_PP_en_v1` converts each chunk into a sparse vector:
- Sparse = mostly zeros, only important token positions have values
- Captures keyword importance with learned term expansion
- "listing" → also activates "exchange", "trading", "market"
- Complementary to dense: catches exact keywords dense misses

**6. Storage in Qdrant**
Each chunk stored as a point with:
```
{
  id: uuid,
  vector: {
    "dense": [768 floats],
    "sparse": {indices: [...], values: [...]}
  },
  payload: {
    text, page_number, paragraph_number, document_id, chunk_index, ...
  }
}
```

**7. Multimodal extraction (PDFs only)**
`unstructured` library detects tables and images.
Each table/image → GPT-4o vision → text summary → embedded → stored as additional chunks.

---

### Step B: Query

**1. HyDE (optional)**
*Hypothetical Document Embeddings* — instead of embedding the raw question, ask GPT-4o to write a short passage that would answer it, then embed that passage.

Example:
- Query: *"Does the fund promote sales to EU public?"*
- Hypothesis: *"The sub-fund is authorised to promote the sale of its units to the public within the EU under UCITS Directive 2009/65/EC..."*
- The hypothesis contains domain keywords that match the relevant chunks better

**2. Dense encoding**
Embed query (or hypothesis) using BGE → 768-dim vector.

**3. Sparse encoding**
Encode query using SPLADE → sparse vector.

**4. Qdrant Hybrid Search with RRF**
Qdrant runs two searches in parallel:
- Dense search: top 40 candidates by cosine similarity
- Sparse search: top 10 candidates by dot product

Then fuses them using **RRF (Reciprocal Rank Fusion)**:
```
score(doc) = 1/(k + rank_in_dense) + 1/(k + rank_in_sparse)
```
where k=60 is a constant. Asymmetric prefetch (40 dense, 10 sparse) weights dense results ~4x more.

**5. Cross-encoder reranking (optional)**
`BAAI/bge-reranker-base` jointly encodes each (query, passage) pair:
- More accurate than bi-encoder similarity
- Too slow for first-stage retrieval (O(n) inference per pair)
- Used as second-stage filter on top candidates

**6. Generation**
Top-K chunks formatted as numbered context:
```
[Source 1] (Page 2, Para 3, relevance: 0.6569):
The Fund is an ETF. The Shares of this Fund are fully transferable...
```
GPT-4o receives the context + question + UCITS-aware system prompt.
The prompt instructs it to:
- Apply UCITS domain knowledge (e.g., UCITS ETF = EU-authorised)
- Answer Yes/No first, then cite evidence
- Always reference [Source X] (Page Y, Para Z)

---

### Step C: Explainability

Every response includes:
- **Sources used**: which chunks, which pages, which paragraphs
- **Relevance scores**: how similar each chunk was to the query
- **Retrieval method**: dense / hybrid_rrf / hyde_dense / hyde_hybrid_rrf
- **Retrieval timings**: how long each stage took
- **Confidence score**: mean retrieval score (0-1)
- **HyDE hypothesis**: the exact text GPT-4o generated as search hypothesis (if HyDE enabled)
- **Faithfulness score**: heuristic measure of answer groundedness
- **Relevancy score**: heuristic measure of answer relevance

---

## 4. Key Terms Glossary

### Retrieval Concepts

**RAG (Retrieval-Augmented Generation)**
A technique where an LLM is given retrieved document context before generating an answer. Prevents hallucination by grounding the answer in real text.

**Dense Retrieval**
Embeds text as a continuous vector (e.g., 768 floats). Captures semantic meaning. "Car" is close to "vehicle" even though they share no characters. Uses bi-encoder architecture.

**Sparse Retrieval (BM25 / SPLADE)**
Represents text as sparse vectors over vocabulary. BM25 is statistical (term frequency). SPLADE is learned — it expands terms semantically while keeping sparse structure. "Promote" might activate "market", "sell", "distribute".

**Hybrid Search**
Combines dense + sparse retrieval results. Dense catches semantic matches, sparse catches exact keyword matches. Together they cover more ground.

**RRF (Reciprocal Rank Fusion)**
A score-free fusion formula: `1/(k + rank)`. Combines ranked lists from multiple retrievers. A document ranked #1 in both lists scores higher than one ranked #1 in only one. No need to normalize scores from different retrievers.

**Asymmetric Prefetch (our approach)**
Dense gets 40 candidates, sparse gets 10. Since RRF weighs by rank (not score), giving dense more candidates means dense results dominate the fused pool. Effectively weights dense ~4x higher.

**Cross-Encoder Reranker**
Takes (query, passage) pairs and scores them jointly. More accurate than bi-encoder similarity because it sees both together. Used as second-stage filter — too slow for first-stage retrieval.

**HyDE (Hypothetical Document Embeddings)**
From Gao et al. (ACL 2023). Instead of embedding the query, generate a hypothetical answer passage and embed that. The hypothesis "bridges the gap" between query language and document language.

*When to use:* When queries use different vocabulary than the answers (e.g., "promote EU sales" vs "shares listed on exchanges").

*When not to use:* When query vocabulary matches document vocabulary well, or when using a non-domain-specific hypothesis generator.

**ColBERT**
Late-interaction model. Embeds query and document at token level, computes max-similarity per query token. More accurate than single-vector bi-encoder but more storage-intensive.

**CRAG (Corrective RAG)**
Evaluates retrieved documents before sending to LLM. Discards irrelevant chunks. Reduces hallucination from bad retrievals.

**GraphRAG**
Builds a knowledge graph from the document. Retrieves via graph traversal. Better for questions that span multiple sections.

---

### Embedding Models

**BGE (BAAI/bge-base-en-v1.5)**
- 768-dimensional dense embeddings
- Trained by Beijing Academy of AI
- Normalized vectors → cosine similarity = dot product
- Good at semantic similarity

**SPLADE (prithivida/Splade_PP_en_v1)**
- Sparse lexical model
- Learns term importance + expansion
- "promote" → also weights "market", "sell", "distribute"
- Implemented via `fastembed`

**BGE-Reranker-Base (BAAI/bge-reranker-base)**
- Cross-encoder architecture
- Takes (query, passage) pair, outputs relevance score
- More accurate than bi-encoder but O(n) per query

---

### Evaluation Metrics

**Exact Match (EM)**
Does the predicted answer exactly match the ground truth (after normalization)?
- `normalize`: lowercase, remove articles/punctuation
- Score: 1.0 or 0.0
- Low for RAG because generated answers are longer than ground truth spans

**Token F1**
Token-level precision and recall between prediction and ground truth.
- `precision = common_tokens / predicted_tokens`
- `recall = common_tokens / truth_tokens`
- F1 = harmonic mean
- Better than EM for partial matches

**ROUGE-L**
Longest Common Subsequence based metric.
- Measures longest matching word sequence
- Used for long-form answer evaluation

**Hit Rate (HR)**
Is the correct answer present in any of the top-K retrieved chunks?
- Binary: 1.0 or 0.0
- Measures recall of the retrieval stage

**MRR (Mean Reciprocal Rank)**
`1 / rank_of_first_relevant_chunk`
- 1.0 if correct chunk is rank 1
- 0.5 if rank 2, 0.33 if rank 3
- Measures ranking quality, not just recall

**Precision@K**
Fraction of top-K chunks that are relevant.
- `relevant_chunks / k`
- Measures precision of retrieval

**Faithfulness**
LLM-judge score: is the answer supported by the retrieved context?
- 1.0 = fully grounded in context
- 0.0 = answer contradicts or ignores context
- Measures hallucination

**Relevancy**
LLM-judge score: does the answer address the question?
- 1.0 = directly answers the question
- 0.0 = off-topic answer

**Correctness**
LLM-judge score: is the answer factually correct vs ground truth?
- 1.0 = fully correct
- 0.0 = wrong

---

### Benchmarks Used

**HotpotQA (distractor setting)**
Multi-hop reasoning QA. Each question needs 2 supporting passages out of 10 (8 are distractors).
Tests: multi-hop retrieval, reasoning over multiple sources.

**SQuAD 2.0**
Short-form extractive QA on Wikipedia passages. Includes unanswerable questions.
Tests: precise span extraction, handling of unanswerable questions.

**BoolQ**
Yes/No questions with a Wikipedia passage as context.
Tests: binary fact verification.

**LLM-as-Judge**
Using GPT-4o to score faithfulness, relevancy, and correctness. More reliable than string matching for generated answers. From Joshi et al. (ICAAIC 2024).

---

### Infrastructure

**Qdrant**
Open-source vector database. Stores vectors and enables fast similarity search. Supports named vectors (dense + sparse in one collection), prefetch, and server-side RRF fusion.

**FastAPI**
Python web framework. Auto-generates Swagger UI at `/docs`. Async support for concurrent requests.

**Docker**
Runs Qdrant as an isolated container. Port 6333 is exposed to the host.

**pydantic-settings**
Reads configuration from `.env` file into typed Python settings. Environment variables override `.env` values.

---

## 5. Evaluation Results & Findings

50 questions per dataset, 4 retrieval modes, with LLM-as-judge scoring.

### Retrieval Quality

| Mode | HotpotQA HR | SQuAD HR | BoolQ HR |
|---|---|---|---|
| dense | **0.80** | 0.42 | 0.08 |
| hybrid | 0.76 | 0.42 | 0.08 |
| hyde_dense | 0.72 | 0.42 | 0.08 |
| hyde_hybrid | 0.66 | 0.42 | 0.08 |

### Generation Quality

| Mode | HotpotQA Faith. | HotpotQA Correct. | BoolQ Correct. |
|---|---|---|---|
| dense | **0.812** | **0.790** | 0.856 |
| hybrid | 0.776 | 0.790 | 0.858 |
| hyde_dense | 0.660 | 0.654 | 0.800 |
| hyde_hybrid | 0.630 | 0.600 | 0.790 |

### Key Findings

**Finding 1: Dense-only wins on standard benchmarks**
Hit Rate: dense (0.80) > hybrid (0.76) > hyde_dense (0.72) > hyde_hybrid (0.66) on HotpotQA.

**Finding 2: SPLADE adds noise on semantic queries**
The query "promote sale to EU public" has zero keyword overlap with the answer "shares fully transferable, listed on exchanges". SPLADE matches the MSCI disclaimer ("NOT PROMOTED BY MSCI") instead — a false positive.

**Finding 3: HyDE is domain-sensitive**
HyDE hurts on general benchmarks because the hypothesis generator was prompted for UCITS financial documents. For Wikipedia QA, the generated hypothesis diverges from Wikipedia passage style, reducing embedding similarity with correct passages.

**Finding 4: HyDE helps on domain-specific queries**
On the UCITS document, HyDE successfully bridged the vocabulary gap between compliance questions and implicit document language.

**Finding 5: EM = 0 everywhere (expected)**
Exact Match requires the generated answer to exactly equal the ground truth span. A generative RAG system produces full explanations, not extracted spans. F1/ROUGE-L are more appropriate but still penalise longer answers.

**Finding 6: BoolQ hit_rate = 0.08 (metric mismatch)**
Ground truth is "True"/"False". These strings don't appear in the retrieved passages. LLM correctness (0.79-0.86) shows the system answers correctly — the text-matching metric cannot capture this.

---

## 6. File Structure

```
RAG/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py           ← Upload + Query endpoints
│   │   ├── core/
│   │   │   └── config.py           ← Settings (reads from .env)
│   │   ├── main.py                 ← FastAPI app factory + startup
│   │   └── services/
│   │       ├── document_loader.py  ← PDF/DOCX/TXT loading + paragraph detection
│   │       └── vector_store.py     ← Qdrant operations (dense+sparse, hybrid_search)
│   ├── evaluation/
│   │   ├── datasets.py             ← HotpotQA, SQuAD, BoolQ loaders
│   │   ├── metrics.py              ← EM, F1, ROUGE-L, Hit Rate, MRR, LLM-judge
│   │   └── runner.py               ← EvaluationRunner (orchestrates evaluation)
│   └── rag/
│       ├── chunking/
│       │   └── chunker.py          ← Semantic + recursive chunking
│       ├── embedding/
│       │   └── embedder.py         ← BGE embedding model (singleton)
│       ├── explainability/
│       │   └── engine.py           ← Confidence scoring, source attribution
│       ├── generation/
│       │   └── generator.py        ← GPT-4o answer generation + prompt templates
│       ├── metrics/
│       │   └── tracker.py          ← Per-query latency + quality metrics
│       ├── multimodal/
│       │   └── processor.py        ← Table/image extraction + GPT-4o summaries
│       └── retrieval/
│           ├── hyde.py             ← HyDE hypothesis generation (NEW)
│           ├── retriever.py        ← HybridRetriever (dense+sparse+reranker+HyDE)
│           └── sparse_encoder.py   ← SPLADE encoding via fastembed
├── data/
│   └── uploads/                    ← Uploaded PDFs stored here
├── evaluation_results/             ← Evaluation JSON + CSV outputs
│   ├── dense/
│   ├── hybrid/
│   ├── hyde_dense/
│   ├── hyde_hybrid/
│   └── mode_comparison_*.json
├── .env                            ← API keys + config (never committed)
├── .gitignore
├── requirements.txt
└── run_evaluation.py               ← Evaluation entry point
```

---

## 7. How to Run

### Prerequisites
1. Docker Desktop running
2. Python 3.11
3. `.env` file with `OPENAI_API_KEY=sk-...`

### Start Qdrant
```bash
docker start qdrant
# First time: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### Start the API server
```bash
cd "C:\Users\Vardhan Reddy Gurram\RAG"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/docs` to use the API.

### Run evaluation
```bash
# Quick test (5 questions, no judge)
python -m run_evaluation --experiment benchmarks --n 5 --no-judge --mode dense

# Compare all modes on your document
python -m run_evaluation --experiment custom --n 10 --mode compare --corpus "data/uploads/your_doc.pdf"

# Full evaluation (50 questions, all modes, with LLM judge)
python -m run_evaluation --experiment both --n 50 --mode compare --corpus "data/uploads/your_doc.pdf"
```

### Configuration (.env)
```
OPENAI_API_KEY=sk-...
QDRANT_HOST=localhost
QDRANT_PORT=6333
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
SPARSE_MODEL=prithivida/Splade_PP_en_v1
RERANKER_MODEL=BAAI/bge-reranker-base
USE_HYBRID=false        # true = dense+sparse RRF
USE_RERANKER=false      # true = cross-encoder reranking
USE_HYDE=true           # true = HyDE hypothesis generation
DENSE_PREFETCH=40       # candidates from dense retriever
SPARSE_PREFETCH=10      # candidates from sparse retriever
```

---

## Research References

| Paper | Used for |
|---|---|
| Lewis et al., "RAG for Knowledge-Intensive NLP Tasks" (NeurIPS 2020) | RAG architecture foundation |
| Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (ACL 2023) | HyDE implementation |
| Formal et al., "SPLADE: Sparse Lexical and Expansion Model" (SIGIR 2021) | Sparse encoding |
| Wang et al., "BGE M3-Embedding" (2024) | Dense embedding model |
| Salemi & Zamani, "Evaluating Retrieval Quality in RAG" (SIGIR 2024) | Evaluation metrics |
| Joshi et al., "Robust Multi-Model RAG Pipeline" (ICAAIC 2024) | Multi-type evaluation, LLM-judge |
| Barnett et al., "Seven Failure Points When Engineering a RAG" (CAIN 2024) | Pipeline design decisions |
