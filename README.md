# Explainable RAG - Information Retrieval Using Explainable Retrieval-Augmented Generation

## Overview
A production-ready RAG (Retrieval-Augmented Generation) system with built-in explainability features. Users can upload documents, ask questions, and receive answers with transparent source attribution, confidence scores, and retrieval explanations.

## Project Structure

```
explainable-rag/
│
├── backend/                    # Python FastAPI backend
│   ├── app/                    # Application layer
│   │   ├── api/routes.py       # API endpoints (upload, query)
│   │   ├── core/config.py      # Settings from .env
│   │   ├── services/           # Vector store, document loader
│   │   └── main.py             # FastAPI entry point
│   ├── rag/                    # RAG pipeline modules
│   │   ├── chunking/           # Document chunking strategies
│   │   ├── embedding/          # Embedding model management
│   │   ├── retrieval/          # Vector search & hybrid retrieval
│   │   ├── generation/         # LLM response generation
│   │   └── explainability/     # Confidence scoring & attribution
│   └── tests/                  # Unit & integration tests
│
├── frontend/                   # React web application (Step 4)
├── docker/                     # Docker configuration files
├── data/uploads/               # Uploaded documents (gitignored)
├── docker-compose.yml          # Orchestration for all services
└── .env                        # Environment variables (gitignored)
```

## Tech Stack
- **Backend:** Python 3.12, FastAPI
- **RAG Pipeline:** LangChain, HuggingFace Transformers
- **Vector Database:** Qdrant (open source, Docker)
- **Embedding Model:** BAAI/bge-base-en-v1.5 (768 dimensions)
- **LLM:** OpenAI GPT-4o
- **Frontend:** React (planned)
- **Deployment:** Docker, AWS (planned)

## Getting Started

### Prerequisites
- Python 3.11+
- Docker Desktop (for Qdrant)
- OpenAI API key

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
2. **Upload a document:** Use `POST /api/v1/upload` → upload your PDF
3. **Ask a question:** Use `POST /api/v1/query` with:
   ```json
   {
     "question": "Is the Fund an exchange-traded fund?",
     "document_id": "abc12345",
     "top_k": 5
   }
   ```
4. **Get answer + explainability:** Response includes the answer, source chunks, confidence score, and retrieval metadata

### Qdrant Dashboard
Visit **http://localhost:6333/dashboard** to see your stored vectors and collections.

## Author
Jaswanth — MSc Data Analytics, Dublin City University
