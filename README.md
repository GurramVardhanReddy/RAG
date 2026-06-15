# Explainable RAG — Information Retrieval Using Explainable Retrieval-Augmented Generation

## Overview

A production-ready **multimodal RAG** system with built-in explainability and **domain-adaptive embedding selection**. Users upload PDF/DOCX documents — the system automatically identifies the document domain, selects the best pre-trained domain-specific embedding model, and provides full XAI attribution explaining why that model was chosen.

**v0.3.0 New Feature: Domain-Adaptive Embedding Selection**
The system now automatically routes documents to domain-specific embedding models (Finance, Legal, Medical, Scientific, Code, E-commerce, or General) and explains the selection using keyword attribution XAI.

---

## Domain → Embedding Model Routing

| Domain | Embedding Model | Dimension |
|---|---|---|
| Finance / Regulatory | `nickmuchi/sec-bert-base` | 768d |
| Legal | `nlpaueb/legal-bert-base-uncased` | 768d |
| Medical / Biomedical | `dmis-lab/biobert-base-cased-v1.2` | 768d |
| Scientific / Academic | `allenai/scibert_scivocab_uncased` | 768d |
| Code / Software | `microsoft/codebert-base` | 768d |
| E-commerce / Product | `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 768d |
| General (fallback) | `BAAI/bge-m3` | 1024d |

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
           │ (7 models)       │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │  SPLADE Sparse  │
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
           │ Dense + Sparse  │
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
│   │   │   └── sparse_encoder.py      # SPLADE sparse encoding
│   │   ├── generation/
│   │   │   └── generator.py           # GPT-4o answer generation
│   │   ├── explainability/
│   │   │   └── engine.py              # Three-layer explainability
│   │   ├── multimodal/
│   │   │   └── processor.py           # Table/image extraction + GPT-4o
│   │   └── metrics/
│   │       └── tracker.py             # Pipeline performance metrics
│   │
│   └── evaluation/
│       └── kilt_indexer.py            # KILT Wikipedia indexing script
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
├── .env                               # Environment variables (API keys)
├── requirements.txt
└── README.md
```

---

## How to Run (Step-by-Step)

### Prerequisites
- Python 3.12
- Docker Desktop installed and running
- OpenAI API key

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
cd C:\Users\HP\Desktop\RAG
venv\Scripts\activate
```

### Step 4 — Start the backend
```powershell
uvicorn backend.app.main:app --reload
```

You should see:
```
Starting Explainable RAG API v0.3.0
Domain-Adaptive Embedding Selection enabled
SPLADE sparse encoder ready.
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
4. Correct embedding model is loaded (e.g. `nickmuchi/sec-bert-base`)
5. Document is chunked and embedded with the domain model
6. Chunks stored in a domain-specific Qdrant collection
7. Upload response includes full **XAI attribution**:
   - Detected domain + confidence score
   - All 7 domain probability scores
   - Top 15 keywords that triggered the decision (with importance scores + context snippets)
   - Human-readable explanation sentence
   - Whether GPT-4o verification was used and its reasoning

## What Happens on Query

1. `document_id` is used to look up the document's domain + embedding model
2. Query is embedded with the **same model** used at index time (vector space consistency)
3. Hybrid search runs in the correct domain-specific Qdrant collection
4. RRF fusion + cross-encoder reranking
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
| `GET` | `/health` | API health + cached models list |
| `GET` | `/docs` | Swagger UI |

---

## Environment Variables (.env)

```env
OPENAI_API_KEY=sk-...

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=explainable_rag

SPARSE_MODEL=prithivida/Splade_PP_en_v1
RERANKER_MODEL=BAAI/bge-reranker-base

USE_HYBRID=true
USE_RERANKER=true
PREFETCH_LIMIT=20

CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
```

---

## Notes on First Upload per Domain

The first time a new domain is encountered (e.g. first Legal document uploaded):
- The domain model downloads from HuggingFace (~400MB for most models)
- This only happens once — the model is cached in memory for all subsequent uploads
- Subsequent uploads for the same domain are fast (model already loaded)

If your internet is slow, pre-download models by running:
```python
from backend.rag.embedding.embedder import get_embedding_model
get_embedding_model("nickmuchi/sec-bert-base")       # Finance
get_embedding_model("nlpaueb/legal-bert-base-uncased")  # Legal
# etc.
```
