# Explainable RAG — Information Retrieval Using Explainable Retrieval-Augmented Generation

## Overview

A production-ready **multimodal RAG** system with built-in explainability for financial document compliance. Users can upload PDF documents containing text, tables, charts, and images — ask questions — and receive answers with transparent source attribution, confidence scores, page/paragraph references, and retrieval explanations.

The system automates answers to **Section 10 UCITS Application Form** questions using Supplement documents, while providing the provenance tracking required for regulatory transparency.

## Architecture

```
                          ┌─────────────────┐
                          │  PDF/DOCX Upload │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │ Document Loader  │
                          │ (PyPDF/fallback) │
                          └────────┬────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
           ┌────────▼────────┐          ┌─────────▼─────────┐
           │ Semantic Chunker│          │ Multimodal Extractor│
           │ + Paragraph     │          │ (unstructured)      │
           │   Detection     │          │ Tables / Images     │
           └────────┬────────┘          └─────────┬──────────┘
                    │                             │
           ┌────────▼────────┐          ┌─────────▼──────────┐
           │ Dense + Sparse  │          │ GPT-4o Summarize   │
           │ Embeddings      │          │ + Dense + Sparse   │
           │ (BGE + SPLADE)  │          │   Embeddings       │
           └────────┬────────┘          └─────────┬──────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────▼────────┐
                          │     Qdrant      │
                          │  (Named Vectors)│
                          └────────┬────────┘
                                   │
               ════════════════════╪═══════════════════════
                            QUERY PIPELINE
                                   │
                          ┌────────▼────────┐
                          │  User Question  │
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
                          │ Explainability  │
                          │ Engine          │
                          │ (confidence,    │
                          │  page/para refs)│
                          └─────────────────┘
```

## Project Structure

```
explainable-rag/
│
├── backend/
│   ├── app/
│   │   ├── api/routes.py              # API endpoints (upload, query, metrics)
│   │   ├── core/config.py             # Settings from .env
│   │   ├── services/
│   │   │   ├── document_loader.py     # PDF/DOCX loading with fallback
│   │   │   └── vector_store.py        # Qdrant operations (named vectors)
│   │   └── main.py                    # FastAPI entry point
│   │
│   ├── rag/
│   │   ├── chunking/
│   │   │   └── chunker.py             # Semantic + recursive chunking
│   │   ├── embedding/
│   │   │   └── embedder.py            # BGE embedding model
│   │   ├── retrieval/
│   │   │   ├── retriever.py           # Hybrid retriever (Dense + Sparse → RRF → Rerank)
│   │   │   └── sparse_encoder.py      # SPLADE sparse encoding
│   │   ├── generation/
│   │   │   └── generator.py           # GPT-4o response generation
│   │   ├── explainability/
│   │   │   └── engine.py              # Confidence scoring + attribution
│   │   ├── multimodal/                # NEW — Multimodal processing
│   │   │   ├── __init__.py
│   │   │   └── processor.py           # Extract tables/images, GPT-4o summaries
│   │   └── metrics/
│   │       └── tracker.py             # Pipeline metrics (5 categories)
│   │
│   ├── evaluation/
│   │   └── runner.py                  # Benchmark evaluation framework
│   └── tests/
│
├── frontend/                          # React web application (planned)
├── docker/
├── data/
│   ├── uploads/                       # Uploaded documents
│   └── extracted_visuals/             # Extracted table/image files
├── metrics_output/                    # Query metrics JSON/CSV exports
├── docker-compose.yml
├── .env
└── README.md
```

## Tech Stack

| Component         | Technology                               |
| ----------------- | ---------------------------------------- |
| Backend           | Python 3.12, FastAPI                     |
| Dense Embeddings  | BAAI/bge-base-en-v1.5 (768d)             |
| Sparse Embeddings | prithivida/Splade_PP_en_v1               |
| Reranker          | BAAI/bge-reranker-base                   |
| Vector Database   | Qdrant (named vectors: dense + sparse)   |
| LLM               | OpenAI GPT-4o                            |
| Multimodal        | unstructured (PDF layout), GPT-4o vision |
| Frontend          | React (planned)                          |
| Deployment        | Docker, AWS (planned)                    |

## Key Features

**Multimodal RAG** — Extracts tables, charts, and images from PDFs using `unstructured`'s hi-res layout analysis. Each visual element is summarized by GPT-4o into a brief, searchable description and embedded alongside text chunks. When a user asks about data in a table or trend in a chart, the system retrieves both text and visual summaries.

**Hybrid Retrieval** — Combines dense semantic search (BGE embeddings) with sparse lexical search (SPLADE) using Reciprocal Rank Fusion, then reranks with a cross-encoder (BGE reranker). This addresses the known limitation that dense retrieval alone is insufficient for full recall (Wang et al., 2021).

**Explainability** — Every answer includes page numbers, paragraph references, confidence scores, and retrieval method metadata. Provenance tracking directly addresses failure points FP2–FP4 of the seven RAG failure modes (Barnett et al., CAIN 2024).

**Robust PDF Loading** — Primary extraction via PyPDFLoader with automatic fallback to `unstructured` when pages return empty (handles scanned PDFs and unusual encodings).

**Comprehensive Metrics** — Five metric categories tracked per query: pipeline performance timing, retrieval quality, response quality, explainability quality, and resource efficiency. Exportable to JSON/CSV for analysis.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop (for Qdrant)
- OpenAI API key
- Poppler (for PDF image extraction on Windows)

### Setup

```bash
# 1. Navigate to project
cd explainable-rag

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install fastapi uvicorn[standard] python-multipart python-dotenv loguru pydantic pydantic-settings
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-experimental langchain-openai langchain-huggingface
pip install pypdf python-docx sentence-transformers rank-bm25
pip install qdrant-client langchain-qdrant
pip install "unstructured[pdf]"
pip install pytest pytest-asyncio httpx

# 4. Copy env file and add your OpenAI key
cp .env.example .env
# Edit .env → set OPENAI_API_KEY=sk-your-key-here

# 5. Start Qdrant (in a separate terminal)
docker run -p 6333:6333 qdrant/qdrant

# 6. Start the backend
uvicorn backend.app.main:app --reload
```

### Usage

1. Open **http://localhost:8000/docs** (Swagger UI)
2. **Upload a document:** `POST /api/v1/upload` → upload your PDF
3. **Ask a question:** `POST /api/v1/query` with:
   ```json
   {
     "question": "Is the Fund an exchange-traded fund?",
     "document_id": "abc12345",
     "top_k": 5
   }
   ```
4. **Response includes:** answer, source chunks with page/paragraph numbers, confidence score, retrieval timings, and full metrics breakdown

### API Endpoints

| Method | Endpoint                 | Description                                    |
| ------ | ------------------------ | ---------------------------------------------- |
| POST   | `/api/v1/upload`         | Upload and process a document (text + visuals) |
| POST   | `/api/v1/query`          | Ask a question with full explainability        |
| GET    | `/api/v1/documents`      | List all processed documents                   |
| GET    | `/api/v1/metrics`        | View aggregated query metrics                  |
| GET    | `/api/v1/metrics/export` | Export metrics to JSON/CSV                     |
| GET    | `/health`                | Health check with vector DB status             |

### Dashboards

- **Swagger UI:** http://localhost:8000/docs
- **Qdrant Dashboard:** http://localhost:6333/dashboard

## Research Context

This system is part of an MSc Data Analytics practicum project at Dublin City University, supervised by Dr. Irina Tal. It sits at the intersection of applied NLP, explainability research, and financial compliance.

### Key Literature

- Lewis et al. — Original RAG framework
- Barnett et al. (CAIN 2024) — Seven failure points in RAG systems
- Joshi et al. (ICAAIC 2024) — Robust multimodal RAG for documents with text, tables, and images
- Wang et al. — BM25 interpolation for dense retrieval
- Khattab & Zaharia — ColBERT late interaction
- Adadi & Berrada — XAI survey (Peeking Inside the Black-Box)
- Salemi & Zamani (SIGIR 2024) — Evaluating retrieval quality in RAG

## Author

**Jaswanth** — MSc Data Analytics, Dublin City University (Student ID: A00052520)
