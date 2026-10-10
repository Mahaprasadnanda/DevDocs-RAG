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

*Note: Semantic retrieval, text chunking, FAISS, embeddings, and LLM integration are NOT yet implemented.*

## Technology Stack

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic
- Pytest
- HTTPX
- BeautifulSoup 4

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
│   │   ├── core/     # Configuration
│   │   ├── ingestion/# Documentation ingestion pipeline
│   │   └── main.py   # Application entry point
│   ├── data/         # Ignored local data (e.g., ingested JSON records)
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

### Health Endpoint

Verify the backend is running by navigating to the health endpoint in your browser or using `Invoke-RestMethod`:
```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/v1/health/
```

### Run Ingestion Pipeline

To run the offline documentation ingestion pipeline and generate local JSON document records:
```powershell
cd backend
python -m app.ingestion.pipeline
```
*Extracted JSON documents will be written to `backend/data/ingested/`.*

### Testing

Run the pytest suite to verify endpoints and ingestion pipeline components (from the `backend` directory):
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

## Milestone Roadmap

- **Milestone 1:** Project Foundation (Completed)
- **Milestone 2:** Documentation Ingestion and Extraction (Completed)
- **Milestone 3:** Document Chunking and Metadata (Upcoming)
