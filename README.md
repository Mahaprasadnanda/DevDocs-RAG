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
- *Note: Generative AI inference (LLMs), chat memory, and frontends are NOT yet implemented.*

## Technology Stack

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic
- Pytest
- HTTPX
- BeautifulSoup 4
- Sentence Transformers
- FAISS (CPU)
- NumPy

### Frontend
- React
- Vite
- TypeScript

## Initial Architecture

```
devdocs-rag/
├── backend/          # FastAPI backend
│   ├── app/
│   │   ├── api/      # API routes
│   │   ├── chunking/ # Text chunking pipeline
│   │   ├── core/     # Configuration
│   │   ├── ingestion/# Documentation ingestion pipeline
│   │   ├── retrieval/# Query vector mapping & Semantic Retrieval
│   │   ├── vector_store/ # FAISS embedding/indexing system
│   │   └── main.py   # Application entry point
│   ├── data/         # Ignored local data (ingested/, chunks/, index/)
│   ├── tests/        # Pytest test suite
│   └── requirements.txt
├── frontend/         # React frontend scaffold
└── README.md
```

## Setup Instructions (Windows PowerShell)

### Backend Setup

1. Create a Python virtual environment:
   ```powershell
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```powershell
   pip install -r backend\requirements.txt
   ```

### Backend Startup

Start the FastAPI development server:
```powershell
cd backend
uvicorn app.main:app --reload
```

### Endpoints

**Health:** Verify the backend is running.
```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/v1/health/
```

**Retrieval (Top-K):** Semantically retrieve top documentation matches.
```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/v1/retrieve/ -Method POST -ContentType "application/json" -Body '{"query": "How do I define a POST endpoint in FastAPI?", "top_k": 3}'
```

### Run Data Pipelines

**Step 1: Documentation Ingestion**
To run the offline documentation ingestion pipeline and generate local JSON document records:
```powershell
cd backend
python -m app.ingestion.pipeline
```
*Extracted JSON documents will be written to `backend/data/ingested/`.*

**Step 2: Text Chunking**
To chunk the ingested documentation into overlapping segments:
```powershell
cd backend
python -m app.chunking.pipeline
```
*Generated chunks will be written to `backend/data/chunks/`.*

**Step 3: Vector Indexing**
To embed chunks numerically and populate a persistent FAISS index for semantic search:
```powershell
cd backend
python -m app.vector_store.pipeline
```
*(This performs a one-time download of the Sentence Transformers model locally). Generated index artifacts are written to `backend/data/index/`.*

### Testing

Run the pytest suite to verify endpoints and data pipeline components (from the `backend` directory):
```powershell
cd backend
python -m pytest tests/
```

### Frontend Startup

Start the frontend development server:
```powershell
cd frontend
npm install
npm run dev
```

## Limitations
- The ingestion pipeline's HTML extraction is tailored primarily for MkDocs Material structures (used by FastAPI official docs). 
- Text is normalized for markdown-like chunking, but formatting inside tables and deep nested HTML might be flattened.
- Section-heading reconstruction binds to the last seen known header in the chunk's content, which works for sequential text but is an approximation rather than a DOM-parsed DOM path.
- Search leverages exact exhaustive inner-product FAISS queries; it is fast for a few thousand vectors but would require migrating to an approximate graph (e.g. HNSW) at millions of vectors.

## Milestone Roadmap

- **Milestone 1:** Project Foundation (Completed)
- **Milestone 2:** Documentation Ingestion and Extraction (Completed)
- **Milestone 3:** Document Chunking and Metadata (Completed)
- **Milestone 4:** Vector Embeddings and FAISS Indexing (Completed)
- **Milestone 5:** Semantic Retrieval and Top-K Search (Completed)
- **Milestone 6:** Generative LLM Integration (Upcoming)
