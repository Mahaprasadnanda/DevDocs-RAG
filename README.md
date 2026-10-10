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

*Note: Semantic retrieval, RAG, citations, debugging, and LLM integration are NOT yet implemented.*

## Technology Stack

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic
- Pytest

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
│   │   └── main.py   # Application entry point
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

### Testing

Run the pytest suite to verify endpoints (from the `backend` directory):
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

## Milestone Roadmap

- **Milestone 1:** Project Foundation (Current)
- **Milestone 2:** Documentation Ingestion and Extraction (Upcoming)
