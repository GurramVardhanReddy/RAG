# Explainable RAG — Information Retrieval Using Explainable Retrieval-Augmented Generation

## Overview

A production-ready **multimodal RAG** system with built-in explainability and **domain-adaptive embedding selection**. Users upload PDF/DOCX documents — the system automatically identifies the document domain, selects the best pre-trained domain-specific embedding model, and provides full XAI attribution explaining why that model was chosen.

**v0.3.0 New Feature: Domain-Adaptive Embedding Selection**
The system now automatically routes documents to domain-specific embedding models (Finance, Legal, Medical, Scientific, Code, E-commerce, or General) and explains the selection using keyword attribution XAI.

---

## Domain → Embedding Model Routing

| Domain | Embedding Model | Dimension |
|---|---|---|
| Finance / Regulatory | `FinLang/finance-embeddings-investopedia` | 768d |
| Legal | `Stern5497/sbert-legal-xlm-roberta-base` | 768d |
| Medical / Biomedical | `BAAI/bge-base-en-v1.5` (see Retrieval Benchmark) | 768d |
| Scientific / Academic | `BAAI/bge-base-en-v1.5` (see Retrieval Benchmark) | 768d |
| Code / Software | `jinaai/jina-embeddings-v2-base-code` | 768d |
| E-commerce / Product | `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 768d |
| General (fallback) | `BAAI/bge-base-en-v1.5` | 768d |

The source of truth is `DOMAIN_MODEL_MAP` in `backend/rag/embedding/domain_detector.py`.

**XAI Method:** Keyword TF-IDF attribution across 7 domain lexicons. For low-confidence cases, GPT-4o verification is used and its reasoning is stored. Every domain decision is fully auditable.

---

## Architecture

```
                          ┌─────────────────────┐
                          │   PDF/DOCX Upload    │
                          └────────┬────────────┘
                                   │
                          ┌────────▼────────────┐
                          │  Document Loader     │
                          │  (PyPDF/fallback)    │
                          └────────┬────────────┘
                                   │
                    ┌──────────────┴──────────────────┐
                    │                                 │
           ┌────────▼────────┐             ┌──────────▼──────────┐
           │ Domain Detector  │             │ Multimodal Extractor │
           │  (NEW v0.3.0)    │             │ (unstructured)       │
           │                  │             │ Tables / Images      │
           │ Keyword TF-IDF   │             └──────────┬──────────┘
           │ attribution XAI  │                        │
           │ → domain label   │             ┌──────────▼──────────┐
           │ → model name     │             │ GPT-4o Summarize    │
           └────────┬────────┘             └──────────┬──────────┘
                    │                                 │
           ┌────────▼────────┐                        │
           │ Domain-Specific  │                        │
           │ Embedding Model  │◄───────────────────────┘
           │ (per domain)     │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │  BM25 Sparse    │
           │  Encoder        │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │  Qdrant         │
           │  (Per-Domain    │
           │   Collections)  │
           └────────┬────────┘
                    │
    ════════════════╪════════════════
              QUERY PIPELINE
                    │
           ┌────────▼────────┐
           │  User Question  │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │ Domain lookup   │  ← same model used at index time
           │ for document_id │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │ Domain Model    │  ← embeds query in same vector space
           │ Query Embed     │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │ Hybrid Retrieval│
           │ Dense + BM25    │
           │  → RRF → Rerank │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │ GPT-4o Generate │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │ Three-Layer     │
           │ Explainability  │
           │ Engine          │
           └─────────────────┘
```

---

## Project Structure

```
explainable-rag/
│
├── backend/
│   ├── app/
│   │   ├── api/routes.py              # API endpoints (upload + domain XAI, query, health)
│   │   ├── core/config.py             # Settings from .env
│   │   ├── services/
│   │   │   ├── document_loader.py     # PDF/DOCX loading with fallback
│   │   │   └── vector_store.py        # Qdrant (per-domain collections)  [UPDATED]
│   │   └── main.py                    # FastAPI entry point               [UPDATED]
│   │
│   ├── rag/
│   │   ├── chunking/
│   │   │   └── chunker.py             # Semantic + recursive chunking
│   │   ├── embedding/
│   │   │   ├── domain_detector.py     # Domain classifier + XAI          [NEW]
│   │   │   └── embedder.py            # Multi-model cache manager         [UPDATED]
│   │   ├── retrieval/
│   │   │   ├── retriever.py           # Domain-aware HybridRetriever      [UPDATED]
│   │   │   ├── bm25_sparse.py         # BM25 sparse vectors (Qdrant IDF modifier)
│   │   │   └── sparse_encoder.py      # Legacy SPLADE encoder (unused, needs fastembed)
│   │   ├── generation/
│   │   │   └── generator.py           # GPT-4o answer generation
│   │   ├── explainability/
│   │   │   └── engine.py              # Three-layer explainability
│   │   ├── multimodal/
│   │   │   └── processor.py           # Table/image extraction + GPT-4o
│   │   └── metrics/
│   │       └── tracker.py             # Pipeline performance metrics
│   │
│   ├── evaluation/
│   │   └── kilt_indexer.py            # KILT Wikipedia indexing script
│   └── tests/                         # pytest: in-memory Qdrant, no model downloads
│
├── scripts/eval_hybrid.py             # BEIR benchmark: dense vs BM25 vs hybrid
├── docker/                            # Backend/frontend Dockerfiles + nginx.conf
├── deploy/AWS.md                      # EC2 + Docker Compose deployment guide
├── .github/workflows/ci.yml           # Lint, tests, image builds
│
├── frontend/                          # React + Vite UI
│   └── src/
│       └── pages/
│           ├── Upload.jsx             # Upload page (shows domain XAI)
│           └── Query.jsx              # Query page + 7 explainability tabs
│
├── data/
│   ├── uploads/                       # Uploaded document files
│   └── extracted_visuals/             # Extracted tables/images
│
├── metrics_output/                    # Per-query metrics JSON logs
├── .env.example                       # Copy to .env (never commit .env)
├── requirements.txt
└── README.md
```

---

## How to Run

### Option A — Full stack with Docker Compose

```bash
cp .env.example .env          # set OPENAI_API_KEY for generation (retrieval works without it)
docker compose up -d --build
curl -s localhost/health
```

The UI is at http://localhost (nginx serves the React build and proxies `/api` and
`/health` to the backend). Qdrant (6333) and the backend (8000) bind to 127.0.0.1 only.
`/health` returns 200 `"healthy"` once the backend reaches Qdrant, and 503 `"degraded"`
until then. Embedding models are cached in the `hf_models` volume, so each domain's
first upload is slow and later ones are fast. For a cloud deployment see
[deploy/AWS.md](deploy/AWS.md).

### Option B — Local development (step by step)

#### Prerequisites
- Python 3.11+
- Docker Desktop installed and running
- OpenAI API key (generation, HyDE and multimodal summaries only)

### Step 1 — Start Docker Desktop
Open Docker Desktop and wait for it to fully start (whale icon in taskbar).

### Step 2 — Start Qdrant (vector database)
Open a terminal and run:
```powershell
docker-compose up -d qdrant
```
This starts Qdrant with a persistent named volume. Qdrant UI available at http://localhost:6333/dashboard.

> **IMPORTANT:** Always use `docker-compose up -d qdrant` and `docker-compose stop` (never `docker-compose down -v`) to preserve your indexed vectors.

### Step 3 — Activate virtual environment
```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4 — Start the backend
```powershell
uvicorn backend.app.main:app --reload
```

You should see:
```
Starting Explainable RAG API v0.3.0
Domain-Adaptive Embedding Selection enabled
Cross-encoder reranker ready.
Explainable RAG API v0.3.0 ready!
```

### Step 5 — Start the frontend (separate terminal)
```powershell
cd frontend
npm run dev
```
Frontend available at http://localhost:3000

---

## What Happens on Upload

1. Document text is extracted
2. **Domain Detector** runs keyword TF-IDF attribution across 7 domain lexicons
3. Domain is identified (e.g. "Finance/Regulatory" for UCITS documents)
4. Correct embedding model is loaded (e.g. `FinLang/finance-embeddings-investopedia`)
5. Document is chunked and embedded with the domain model, plus a BM25 sparse vector per chunk
6. Chunks stored in a domain-specific Qdrant collection (named vectors `dense` and `bm25`)
7. Upload response includes full **XAI attribution**:
   - Detected domain + confidence score
   - All 7 domain probability scores
   - Top 15 keywords that triggered the decision (with importance scores + context snippets)
   - Human-readable explanation sentence
   - Whether GPT-4o verification was used and its reasoning

## What Happens on Query

1. `document_id` is used to look up the document's domain + embedding model
2. Query is embedded with the **same model** used at index time (vector space consistency)
3. Hybrid search runs in the correct domain-specific Qdrant collection: dense
   (40 candidates) and BM25 (10 candidates) arms, with Qdrant applying the IDF
   weighting server-side
4. RRF fusion + cross-encoder reranking. Collections created before the BM25 arm
   have no `bm25` vector; they are detected at startup and served dense-only
   (re-upload to upgrade them)
5. GPT-4o generates the answer
6. Three-layer explainability analysis runs
7. Query response includes `retrieval_timings.domain` and `retrieval_timings.query_embedding_model`

---

## XAI Methods in the System

| Location | XAI Method | What it explains |
|---|---|---|
| **Upload** | Keyword TF-IDF Attribution | Why a domain was detected + why a model was selected |
| **Upload (low confidence)** | GPT-4o reasoning | Natural language justification when keywords are ambiguous |
| **Query — Layer 1** | Source Attribution Precision/Recall | How accurately sources are cited |
| **Query — Layer 1** | Confidence Calibration | How trustworthy the confidence score is |
| **Query — Layer 2** | Chunk Relevance Heatmap | Which retrieved chunks are most relevant |
| **Query — Layer 2** | Provenance Trace | Full audit trail: query → chunks → answer |
| **Query — LIME** | Retrieval LIME | Which query words drove retrieval (word-level importance) |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/upload` | Upload document (returns domain XAI) |
| `POST` | `/api/v1/query` | Query a document |
| `GET` | `/api/v1/documents` | List uploaded documents with domain info |
| `GET` | `/api/v1/health/collections` | Status of all domain Qdrant collections |
| `GET` | `/health` | API health + cached models list (503 when Qdrant is unreachable) |
| `GET` | `/docs` | Swagger UI |

---

## Environment Variables (.env)

Copy `.env.example` to `.env`. The main settings:

```env
OPENAI_API_KEY=            # generation, HyDE, multimodal summaries

QDRANT_HOST=localhost      # "qdrant" inside docker compose (set by compose)
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=explainable_rag

RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_THRESHOLD=0.1

USE_HYBRID=true            # dense + BM25, fused with RRF
USE_RERANKER=true
USE_HYDE=false
DENSE_PREFETCH=40
SPARSE_PREFETCH=10
TOP_K=5
```

---

## Retrieval Benchmark (BEIR)

`scripts/eval_hybrid.py` indexes two BEIR test sets through the real
`VectorStoreService` and `HybridRetriever` (in-memory Qdrant, reranker off) and
scores each retrieval mode against the qrels. Hybrid is dense top-40 + BM25
top-10 fused with RRF. Δ is the paired bootstrap of hybrid − dense nDCG@10
(95% CI). Raw results: `evaluation_results/hybrid_bm25_eval.json`.

| Dataset (queries) | Embedding model | Dense | BM25 | Hybrid | Δ nDCG@10 (95% CI) |
|---|---|---|---|---|---|
| SciFact (300) | `allenai/scibert_scivocab_uncased` (former Scientific route) | 0.146 | 0.667 | 0.536 | +0.390 [+0.352, +0.427] |
| SciFact (300) | `BAAI/bge-base-en-v1.5` (General; Scientific route now) | 0.739 | 0.667 | 0.754 | +0.015 [−0.004, +0.034] |
| NFCorpus (323) | `NeuML/pubmedbert-base-embeddings` (former Medical route) | 0.266 | 0.310 | 0.316 | +0.050 [+0.036, +0.065] |
| NFCorpus (323) | `BAAI/bge-base-en-v1.5` (General; Medical route now) | 0.370 | 0.310 | 0.367 | −0.003 [−0.013, +0.008] |

nDCG@10 in every cell except the last column. Recall@10 follows the same pattern
(e.g. SciFact/SciBERT 0.215 → 0.744).

What this shows:

- **Hybrid significantly improves the weak domain models** (+0.39 nDCG@10 on
  SciFact with SciBERT, +0.05 on NFCorpus with PubMedBERT; both CIs exclude 0).
  With a strong general embedder it is within noise.
- **SciBERT is not a retrieval model.** It is a masked-language model with mean
  pooling, and dense SciBERT scores 0.146 where `bge-base-en-v1.5` scores 0.739.
  On both datasets the General model beat the domain-routed model, so the
  Scientific and Medical routes now use `bge-base-en-v1.5`. For the shipped
  hybrid configuration that moves nDCG@10 from 0.536 to 0.754 on SciFact and
  from 0.316 to 0.367 on NFCorpus (the rows above; no separate run).
  Documents indexed before this change keep the model they were indexed with,
  because queries embed with the model recorded for each document.
- Latency is not reported as a headline: these runs use Qdrant's in-process
  local mode on CPU, which does not represent a Qdrant server.

Reproduce (downloads the datasets and models; takes about 2 hours on CPU):

```bash
python -u scripts/eval_hybrid.py
# the former routes, as in the table:
python -u scripts/eval_hybrid.py --datasets scifact --models allenai/scibert_scivocab_uncased
python -u scripts/eval_hybrid.py --datasets nfcorpus --models NeuML/pubmedbert-base-embeddings
```

---

## Tests and CI

```bash
pytest backend/tests -q
ruff check backend scripts --select E9,F63,F7,F82
```

Tests use an in-memory Qdrant and a fake embedder, so they need no models, no
network and no OpenAI key. GitHub Actions (`.github/workflows/ci.yml`) runs the
lint and tests on every push and pull request, then builds both Docker images.

---

## Notes on First Upload per Domain

The first time a new domain is encountered (e.g. first Legal document uploaded):
- The domain model downloads from HuggingFace (~400MB for most models)
- This only happens once — the model is cached in memory for all subsequent uploads
- Subsequent uploads for the same domain are fast (model already loaded)

If your internet is slow, pre-download models by running:
```python
from backend.rag.embedding.embedder import get_embedding_model
get_embedding_model("FinLang/finance-embeddings-investopedia")  # Finance
get_embedding_model("Stern5497/sbert-legal-xlm-roberta-base")   # Legal
# etc.
```
