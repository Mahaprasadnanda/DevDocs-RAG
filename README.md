# DevDocs RAG

DevDocs RAG is a full-stack technical documentation and debugging assistant.

## Objective

This project aims to use Retrieval-Augmented Generation (RAG) to answer developer questions from curated official documentation. It will retrieve semantically relevant passages using Sentence Transformers and FAISS, generate grounded answers via OpenRouter, and provide an error-aware debugging mode.

## Current Implementation Status

**Milestone 1 completed:** Project Foundation
- Established initial clean modular-monolith structure.
- Configured backend FastAPI application with health check endpoint.
- Configured frontend React, Vite, and TypeScript scaffolding.
- Verified local development environment setup.
- Setup testing and `.gitignore`.

**Milestone 2 completed:** Documentation Ingestion and Extraction
- Maintains a curated list of official FastAPI documentation URLs.
- Fetches HTML pages over HTTP with HTTPX, utilizing configured timeouts and response size limits.
- Validates HTTP responses and captures network/HTML extraction failures gracefully without crashing.
- Parses HTML using BeautifulSoup 4 to extract the main article content, removing navigation, sidebars, and scripts.
- Persists structured JSON document records (titles, headings, cleaned text, source URLs) locally.

**Milestone 3 completed:** Text Chunking and Metadata Preservation
- Reads the ingested JSON documentation records.
- Deterministically splits documents into overlapping chunks (default 700 chars, 100 char overlap).
- Intelligently chunks text by searching for natural boundaries (paragraphs, sentences, words) before falling back to hard splits.
- Preserves all crucial source metadata per chunk (document ID, section headings, URL, canonical URL).
- Persists chunk records individually to prepare for embeddings.

**Milestone 4 completed:** Embeddings and Persistent FAISS Index
- Generates 384-dimensional dense vector embeddings for chunks using `sentence-transformers/all-MiniLM-L6-v2`.
- Text embeddings represent semantic meaning numerically, allowing similarity comparisons.
- Employs batch processing and float32 mathematical normalizations to ensure vector scale consistency (unit length).
- Indexes vectors exactly using FAISS `IndexFlatIP`. Normalized vectors allow inner product to correspond directly to cosine similarity search, avoiding expensive metric calculations.
- Persists the searchable binary FAISS index alongside a consistent JSON metadata manifest linking each vector position strictly back to its source chunk metadata.

**Milestone 5 completed:** Semantic Retrieval and Top-K Search
- Validates queries and expected capacities dynamically (e.g., whitespace constraints, FAISS capacity bounds).
- Embeds natural-language queries uniformly with the index using `all-MiniLM-L6-v2`.
- Executes lightning-fast exhaustive `IndexFlatIP` dot-product mapping (Cosine Similarity) yielding top-K similarity boundaries.
- Links search vector coordinates recursively back into source `ChunkRecord` metadata, providing clean HTTP endpoints exposing `section_title`, `score`, and chunk identifiers natively.
- Prevents redundant model/index cold-loading utilizing API lifecycle boundaries, optimizing inference memory.

**Milestone 6 completed:** Groq LLM Integration, Grounded Answer Generation, Frontend Connection, and Docker Deployment
- Integrated official Groq client (`GroqLLMClient`) supporting `llama-3.3-70b-versatile` by default with structured error handling for configuration, timeout, and provider errors.
- Built a bounded context assembler (`build_context`) that trims retrieved chunks within character budgets while preserving chunk-source mappings (`[S1]`, `[S2]`, etc.).
- Designed a grounded RAG generation pipeline (`RAGService`) with strict anti-hallucination prompts requiring citations and transparent fallback when documentation does not contain answers.
- Implemented `POST /api/v1/ask/` API endpoint returning structured answers, citation references, retrieved context excerpts, and latency metrics.
- Developed an interactive React + TypeScript UI with search input, loading indicators, code block formatting, citation badges, clickable source links, and clear error banners.
- Containerized both backend and frontend using Docker, multi-stage builds, Nginx reverse proxy, and Docker Compose orchestration with automated healthchecks.

## Technology Stack

### Backend
- Python 3.12+
- FastAPI & Uvicorn
- Groq SDK (`groq`)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS (CPU `IndexFlatIP`)
- Pydantic & Pydantic-Settings
- Pytest & HTTPX
- BeautifulSoup 4 & NumPy

### Frontend
- React 18 & TypeScript
- Vite
- Lucide React (Icons)

### Deployment & Infrastructure
- Docker & Docker Compose
- Nginx (Alpine) as frontend web server & reverse proxy

## Architecture

```
devdocs-rag/
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── api/routes/       # API endpoints (health, retrieve, ask)
│   │   ├── chunking/         # Text chunking pipeline
│   │   ├── core/             # Configuration & settings
│   │   ├── ingestion/        # Documentation ingestion pipeline
│   │   ├── llm/              # Groq LLM client wrapper & exceptions
│   │   ├── rag/              # RAG orchestration, context builder & prompts
│   │   ├── retrieval/        # Query embedding & semantic retrieval
│   │   ├── vector_store/     # FAISS embedding/indexing system
│   │   └── main.py           # Application entry point & lifespan manager
│   ├── data/                 # Local data artifacts (ingested/, chunks/, index/)
│   ├── tests/                # Comprehensive test suite (77 tests across M1-M6)
│   ├── Dockerfile            # Container definition for FastAPI backend
│   └── requirements.txt
├── frontend/                 # React + TypeScript frontend
│   ├── src/                  # Application components, styles, types
│   ├── nginx.conf            # Nginx config forwarding /api/ to backend
│   ├── Dockerfile            # Multi-stage production container build
│   └── package.json
├── docker-compose.yml        # Orchestration for frontend & backend services
├── .env.example              # Environment variables template
└── README.md
```

## Setup & Running Instructions

### Option 1: Running with Docker Compose (Recommended)

1. **Configure Environment:**
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Add your Groq API key in `.env`:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   ```
   *(Note: The application will start and serve retrieval queries even without an API key; answer generation requires a valid key.)*

2. **Start Services:**
   ```bash
   docker compose up -d
   ```

3. **Access Services:**
   - **Frontend UI:** [http://localhost:3000](http://localhost:3000)
   - **Backend API & Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Backend Health Check:** [http://localhost:8000/api/v1/health/](http://localhost:8000/api/v1/health/)

4. **Stop Services:**
   ```bash
   docker compose down
   ```

---

### Option 2: Running Locally (Windows PowerShell)

#### Backend Setup

1. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r backend\requirements.txt
   ```

3. Configure environment variables in `backend/.env` or PowerShell:
   ```powershell
   $env:GROQ_API_KEY="gsk_your_groq_api_key"
   ```

4. Start FastAPI server:
   ```powershell
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```

#### Frontend Setup

1. Install frontend dependencies and run Vite dev server:
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
   The dev server will proxy `/api` requests to `http://localhost:8000`.

---

## API Endpoints

### 1. Health Check
```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/v1/health/
```

### 2. Semantic Retrieval (Top-K Search)
```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/v1/retrieve/ `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query": "How do I define path parameters in FastAPI?", "top_k": 3}'
```

### 3. Grounded Question Answering (RAG)
```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/v1/ask/ `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query": "How do path parameters work in FastAPI?", "top_k": 3}'
```

---

## Running Data Pipelines

**Step 1: Documentation Ingestion**
```powershell
cd backend
python -m app.ingestion.pipeline
```
*Extracted JSON documents are written to `backend/data/ingested/`.*

**Step 2: Text Chunking**
```powershell
cd backend
python -m app.chunking.pipeline
```
*Generated chunks are written to `backend/data/chunks/`.*

**Step 3: Vector Indexing**
```powershell
cd backend
python -m app.vector_store.pipeline
```
*Generated FAISS index and metadata manifest are written to `backend/data/index/`.*

---

## Automated Testing

Run the full pytest test suite (77 unit and integration tests covering Milestones 1–6):
```powershell
cd backend
python -m pytest tests/ -v
```

---

## Limitations

- Grounded generation relies strictly on retrieved documentation chunks. If official documentation does not cover the question, the assistant explicitly states it cannot answer based on retrieved context.
- Indexing currently uses exact CPU inner-product search (`IndexFlatIP`). While ultra-fast for hundreds/thousands of documentation chunks, expanding to millions would require an approximate nearest-neighbor index (e.g., FAISS HNSW or IVF).
- LLM response generation requires an active internet connection and valid Groq API key.

## Milestone Roadmap

- **Milestone 1:** Project Foundation (Completed)
- **Milestone 2:** Documentation Ingestion and Extraction (Completed)
- **Milestone 3:** Document Chunking and Metadata (Completed)
- **Milestone 4:** Vector Embeddings and FAISS Indexing (Completed)
- **Milestone 5:** Semantic Retrieval and Top-K Search (Completed)
- **Milestone 6:** Groq LLM Integration, Grounded Answer Generation, Frontend Connection, and Docker Deployment (Completed)

