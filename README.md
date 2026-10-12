# DevDocs RAG — Documentation-Grounded Developer Assistant

DevDocs RAG is a full-stack, production-grade technical documentation and debugging assistant. It retrieves semantically relevant passages from curated official developer documentation, generates grounded answers and structured error diagnoses with verified citations, and provides strict abstention and anti-hallucination safeguards.

---

## 1. System Architecture

The following diagram illustrates the complete end-to-end architecture, from offline documentation ingestion and indexing to online retrieval, Groq LLM generation, citation validation, and user interaction.

```mermaid
flowchart TD
    subgraph Ingestion["Offline Pipeline: Ingestion & Indexing"]
        A["Official Documentation URLs (FastAPI)"] --> B["HTTPX Fetcher (Timeouts, Size Limits, Validations)"]
        B --> C["BS4 Article Extractor (Clean Text, Headings)"]
        C --> D["Document Records (JSON Storage)"]
        D --> E["Text Chunker (700 chars, 100 overlap, Boundary-aware)"]
        E --> F["Chunk Records (Metadata & Document IDs)"]
        F --> G["Embedding Service (all-MiniLM-L6-v2, Normalized)"]
        G --> H[("FAISS Vector Index (IndexFlatIP) + Metadata Manifest")]
    end

    subgraph OnlineRetrieval["Online Retrieval & Relevance Filtering"]
        Q["User Query / Stack Trace"] --> QE["Query Embedder (all-MiniLM-L6-v2)"]
        QE --> FS["FAISS Cosine Similarity Search (Dot Product)"]
        H -.-> FS
        FS --> RF{"Relevance Score >= Threshold (0.30)?"}
        RF -- "Below Threshold" --> AB["Abstain (Safe Fallback Response)"]
        RF -- "Above Threshold" --> CA["Context Assembler (Budget-bounded [S1], [S2]...)"]
    end

    subgraph LLMOrchestration["LLM Orchestration & Verification"]
        CA --> GP["Prompt Construction (System, Guardrails, Context)"]
        GP --> GL["Groq LLM Client (OpenAI-compatible Chat API)"]
        GL --> RP["Raw Completion Response"]
        RP --> CV{"Citation & Schema Validation"}
        CV -- "Invalid Citations / Malformed JSON" --> CR["Corrective Retry (1 retry with feedback)"]
        CR --> GL
        CV -- "Valid Citations" --> SC["Syntactically Verified Response"]
        CR -- "Retry Exhausted" --> FB["Deterministic Safe Fallback"]
    end

    subgraph UserInterface["User Interface (React + Vite + TypeScript)"]
        SC --> FE["Interactive Frontend"]
        FB --> FE
        AB --> FE
        FE --> M1["Ask DevDocs Mode (Q&A with Badges & Source Cards)"]
        FE --> M2["Debug Assistant Mode (Structured Root Cause, Fix & Verification)"]
    end
```

---

## 2. Milestone Summary

- **Milestone 1: Project Foundation**
  - Established modular monolith architecture with FastAPI backend and React + TypeScript frontend.
  - Configured health endpoint, development environment, and initial pytest setup.

- **Milestone 2: Documentation Ingestion and Extraction**
  - Robust HTML extraction using HTTPX and BeautifulSoup 4.
  - Stripped non-content DOM elements, extracted hierarchical headings, and saved structured JSON records.

- **Milestone 3: Text Chunking and Metadata Preservation**
  - Boundary-aware text chunker with paragraph and sentence boundary preservation.
  - Maintained complete metadata lineage (document IDs, section headings, URLs, canonical URLs).

- **Milestone 4: Embeddings and Persistent FAISS Index**
  - Dense 384-dimensional embeddings generated with `sentence-transformers/all-MiniLM-L6-v2`.
  - Normalized L2 vectors indexed with FAISS `IndexFlatIP` for exact inner-product cosine similarity.
  - Persisted binary index alongside metadata manifest.

- **Milestone 5: Semantic Retrieval and Top-K Search**
  - Semantic query search with configurable relevance thresholding (`min_score = 0.30`).
  - Top-K result mapping back to chunk metadata with score and section identifiers.

- **Milestone 6: Groq LLM Integration, Grounded Answer Generation, and Docker Deployment**
  - Official Groq SDK client wrapper with structured error handling.
  - Context assembler bounding prompts within character budgets while preserving chunk-source mappings (`[S1]`, `[S2]`).
  - RAG service implementing strict anti-hallucination guardrails and containerized Docker Compose deployment.

- **Milestone 7: RAG Reliability and Citation Validation**
  - Two-stage citation validation enforcing referential integrity (checking that every citation token refers to a retrieved chunk).
  - Single corrective retry mechanism with exact error diagnostic feedback.
  - Deterministic fallback when citations fail validation or retries exhaust.

- **Milestone 8: Documentation-Grounded Debugging Assistant**
  - Dedicated endpoint `POST /api/v1/debug/` for developer error messages and stack traces.
  - Structured diagnosis schema: `root_cause`, `suggested_fix`, `code_example`, `verification_steps`, and validated citations.
  - Schema retry and safe fallback when LLM output violates JSON formatting.

- **Milestone 9: Full-Stack Integration and UX Hardening**
  - Unified tabbed React frontend supporting both **Ask DevDocs** and **Debug Assistant** modes.
  - Grounded source cards with similarity score badges, section headings, and direct documentation links.
  - Input validation, resilient error banners, and full Docker Compose verification.

- **Milestone 10: Final Verification, Retrieval Evaluation, and Release Readiness**
  - Corpus-grounded 32-query retrieval evaluation benchmark across 6 query categories.
  - Deterministic evaluation test suite verifying MRR, Recall@K, and Abstention metrics.
  - End-to-end performance latency profiling (embedding vs. FAISS search latency).
  - Production verification across all 151 backend tests, TypeScript build, and linter.

---

## 3. Technology Stack

### Backend
- **Framework:** FastAPI (Python 3.12+) with Uvicorn ASGI server
- **LLM Provider:** Groq SDK (`groq`) using `openai/gpt-oss-120b` or `llama-3.3-70b-versatile`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Vector Index:** FAISS (`IndexFlatIP`, CPU-optimized exact inner-product)
- **Data Validation:** Pydantic v2 & Pydantic-Settings
- **HTML Parsing:** BeautifulSoup 4 & HTTPX
- **Testing:** Pytest (151 unit and integration tests)

### Frontend
- **Framework:** React 18 with TypeScript
- **Build Tool:** Vite
- **Styling:** Modular CSS with responsive layout
- **Icons:** Lucide React
- **Linter:** Oxlint (0 warnings, 0 errors)

### Infrastructure & Deployment
- **Containerization:** Docker & Docker Compose
- **Web Server:** Nginx (Alpine) acting as static host and reverse proxy (`/api/` -> FastAPI)
- **Health Checks:** Native Docker Compose healthcheck probes on `/api/v1/health/`

---

## 4. API Endpoints

### 1. Health Check
```http
GET /api/v1/health/
```
**Response (200 OK):**
```json
{
  "status": "ok",
  "message": "API is running",
  "version": "1.0.0",
  "environment": "development"
}
```

### 2. Semantic Retrieval (Top-K Search)
```http
POST /api/v1/retrieve/
Content-Type: application/json
```
**Request Body:**
```json
{
  "query": "How do I define path parameters in FastAPI?",
  "top_k": 3,
  "min_score": 0.3
}
```
**Response (200 OK):**
```json
{
  "query": "How do I define path parameters in FastAPI?",
  "count": 3,
  "results": [
    {
      "chunk_id": "6634008cc0069f6528f5c6a79d0b3f98169d3156a378c4336e9c96c626251455_0",
      "text": "Path Parameters\nYou can declare path \"parameters\" or \"variables\" with the same syntax used by Python format-strings...",
      "source_url": "https://fastapi.tiangolo.com/tutorial/path-params/",
      "document_id": "6634008cc0069f6528f5c6a79d0b3f98169d3156a378c4336e9c96c626251455",
      "section_title": "Path Parameters",
      "score": 0.7421
    }
  ]
}
```

### 3. Grounded Question Answering (RAG)
```http
POST /api/v1/ask/
Content-Type: application/json
```
**Request Body:**
```json
{
  "query": "How do path parameters work with Python type hints in FastAPI?",
  "top_k": 5
}
```
**Response (200 OK):**
```json
{
  "query": "How do path parameters work with Python type hints in FastAPI?",
  "answer": "You can declare path parameters with Python type annotations [S1]. FastAPI uses these type declarations to provide data parsing, data validation, and automatic API documentation [S1].",
  "citations": [
    {
      "source_id": "S1",
      "url": "https://fastapi.tiangolo.com/tutorial/path-params/",
      "title": "Path Parameters",
      "chunk_id": "6634008cc0069f6528f5c6a79d0b3f98169d3156a378c4336e9c96c626251455_0",
      "score": 0.7421
    }
  ],
  "retrieved_context": [ ... ],
  "latency_ms": 1150.2
}
```

### 4. Documentation-Grounded Debug Assistant
```http
POST /api/v1/debug/
Content-Type: application/json
```
**Request Body:**
```json
{
  "error_message": "fastapi.exceptions.HTTPException: 404: Item not found",
  "context": "Raised inside a path operation when item is not in database",
  "framework": "fastapi",
  "top_k": 5
}
```
**Response (200 OK):**
```json
{
  "root_cause": "An HTTPException was raised to indicate an HTTP error response to the client [S1].",
  "suggested_fix": "Catch and raise HTTPException with the appropriate status_code and detail message as documented [S1].",
  "code_example": "from fastapi import FastAPI, HTTPException\n\napp = FastAPI()\n\n@app.get(\"/items/{item_id}\")\nasync def read_item(item_id: str):\n    if item_id not in items:\n        raise HTTPException(status_code=404, detail=\"Item not found\")",
  "verification_steps": [
    "Run uvicorn and request an invalid item ID.",
    "Verify the JSON response returns status code 404 with {\"detail\": \"Item not found\"}."
  ],
  "citations": [
    {
      "source_id": "S1",
      "url": "https://fastapi.tiangolo.com/tutorial/handling-errors/",
      "title": "Handling Errors",
      "chunk_id": "ad230cc28bf3ccef2b9fd9fde243db6117503c95e4adcac15f6d26397a79382a_0",
      "score": 0.718
    }
  ],
  "latency_ms": 1420.5
}
```

---

## 5. Retrieval Evaluation & Quality Benchmark

### Evaluation Methodology
The evaluation suite evaluates semantic retrieval across a curated, corpus-grounded benchmark consisting of **32 test queries** derived strictly from the 5 indexed FastAPI documentation pages (plus out-of-domain queries):
- **Factual queries (6):** Specific configuration commands, defaults, and requirements.
- **API usage queries (6):** How-to questions regarding decorators, exception handlers, and request models.
- **Parameter queries (5):** Inquiries regarding function parameters, type hints, and status codes.
- **Multi-detail queries (5):** Complex questions combining validation, documentation, and error handling.
- **Similar-sounding distinctions (5):** Disambiguating path parameters vs body vs exceptions.
- **Out-of-domain queries (5):** Queries outside the corpus (e.g., Quantum Chromodynamics, Django ORM, React hooks) to verify abstention.

### Metric Definitions
- **Recall@K (Document-Level):** The proportion of in-domain queries for which at least one chunk from the expected ground-truth document is retrieved within the top $K$ results. Document-level ranking collapses multiple chunks from the same document to prevent multi-chunk inflation.
- **Mean Reciprocal Rank (MRR):** The average of the reciprocal ranks of the first relevant document across all in-domain queries:
  $$\text{MRR} = \frac{1}{|Q_{\text{in}}|} \sum_{i=1}^{|Q_{\text{in}}|} \frac{1}{\text{rank}_i}$$
- **Abstention Filter Rate:** The percentage of out-of-domain queries that correctly return 0 retrieved chunks when scored against the relevance threshold (`min_score = 0.30`).

### Actual Benchmark Results

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Evaluated Queries** | **32 total** | 27 in-domain, 5 out-of-domain queries |
| **Recall@1** | **70.37%** (19/27) | Expected document returned at rank 1 |
| **Recall@3** | **96.30%** (26/27) | Expected document returned in top 3 |
| **Recall@5** | **100.00%** (27/27) | Expected document returned in top 5 |
| **Mean Reciprocal Rank (MRR)** | **0.8302** | High average rank for primary document |
| **Abstention Filter Rate** | **100.00%** (5/5) | Complete filtering of out-of-domain questions |

### Latency & Performance Breakdown

| Stage | Median Latency (p50) | 95th Percentile Latency (p95) |
| :--- | :--- | :--- |
| **End-to-End Retrieval** | **14.57 ms** | **17.55 ms** |
| **Query Embedding Inference** | **15.76 ms** | **20.84 ms** |
| **FAISS Vector Search** | **0.04 ms** | **0.08 ms** |

### Index Storage Metrics
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Dimension:** 384 float32
- **Indexed Vectors:** 99 vectors across 5 FastAPI documentation pages
- **Total Index Disk Footprint:** 261.9 KB (`vector.index`: 152.1 KB, `metadata.json`: 115.9 KB, `manifest.json`: 152 B)

---

## 6. Citation Verification vs. Semantic Claim Entailment

It is vital to distinguish between what the system validates and what requires human review:

> [!IMPORTANT]
> **Syntactic Citation Validation (Enforced by DevDocs RAG):**
> 1. **Referential Integrity:** Verifies that every bracketed citation token (e.g., `[S1]`, `[S2]`) in the LLM's response maps strictly to an actual retrieved chunk in the prompt's context.
> 2. **URL Integrity:** Verifies that every cited source URL matches the real canonical URL from the metadata manifest, preventing lookalike or fabricated links.
> 3. **Non-Empty Sourcing:** Rejects answers with missing or fabricated source references.
>
> **Semantic Entailment Limitation (Requires User Review):**
> Syntactic validation ensures that cited sources exist and were provided to the model. However, it does not mathematically prove Natural Language Entailment (NLE) for every clause. While strict system prompts and low temperature minimize drift, developers should always click the cited source cards to review the official documentation for mission-critical production changes.

---

## 7. Setup & Running Instructions

### Option 1: Running with Docker Compose (Recommended)

1. **Configure Environment:**
   ```bash
   cp .env.example .env
   ```
   Add your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   ```

2. **Start Services:**
   ```bash
   docker compose up -d
   ```

3. **Access Services:**
   - **Frontend UI:** [http://localhost:3000](http://localhost:3000)
   - **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Health Endpoint:** [http://localhost:8000/api/v1/health/](http://localhost:8000/api/v1/health/)

4. **Stop Services:**
   ```bash
   docker compose down
   ```

---

### Option 2: Running Locally (Windows PowerShell)

#### Backend Setup
```powershell
# 1. Activate Python virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r backend\requirements.txt

# 3. Configure environment
$env:GROQ_API_KEY="gsk_your_groq_api_key"

# 4. Start FastAPI server
cd backend
uvicorn app.main:app --reload --port 8000
```

#### Frontend Setup
```powershell
cd frontend
npm install
npm run dev
```

---

## 8. Verification & Test Execution

### Running Backend Unit & Regression Tests (151 Tests)
```powershell
cd backend
python -m pytest tests/ -v
```

### Running Retrieval Benchmark Evaluation
```powershell
cd backend
python scripts/evaluate_retrieval.py
```
*Results are written to `backend/data/eval/retrieval_eval_results.json`.*

### Building and Linting Frontend
```powershell
cd frontend
npm run lint    # Oxlint check (0 errors, 0 warnings)
npm run build   # TypeScript compilation & Vite bundle
```

---

## 9. Release Readiness Assessment

| Verification Area | Status | Evidence |
| :--- | :--- | :--- |
| **Backend Test Suite** | Passed | 151 / 151 tests passing in pytest |
| **Frontend TypeScript** | Passed | `tsc -b` compiled cleanly |
| **Frontend Linter** | Passed | Oxlint verified with 0 errors, 0 warnings |
| **Retrieval Recall@5** | Passed | 100.00% across in-domain test queries |
| **Abstention Quality** | Passed | 100.00% abstention on out-of-domain queries |
| **Docker Compose** | Healthy | Both `devdocs-backend` and `devdocs-frontend` running and passing healthchecks |
| **Security & Secrets** | Verified | No API keys committed; clean `.env.example` template |
