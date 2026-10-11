import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.rag.models import AskResponse, SourceCitation
from app.retrieval.service import IndexUnavailableError
from app.llm.client import GroqConfigurationError, GroqTimeoutError, GroqProviderError

client = TestClient(app)

@patch("app.api.routes.ask.rag_service")
def test_ask_endpoint_success(mock_rag):
    mock_rag.answer.return_value = AskResponse(
        query="How to create a POST endpoint?",
        answer="Use the `@app.post()` decorator in FastAPI [S1].",
        sources=[
            SourceCitation(
                source_id="S1",
                chunk_id="chunk_123",
                document_id="doc_456",
                section_title="First Steps",
                source_url="https://fastapi.tiangolo.com/tutorial/first-steps/",
                score=0.91
            )
        ],
        total_sources=1,
        model="llama-3.3-70b-versatile"
    )

    resp = client.post("/api/v1/ask/", json={"query": "How to create a POST endpoint?", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "How to create a POST endpoint?"
    assert "@app.post()" in data["answer"]
    assert data["total_sources"] == 1
    assert data["sources"][0]["source_id"] == "S1"
    assert data["sources"][0]["chunk_id"] == "chunk_123"
    assert data["model"] == "llama-3.3-70b-versatile"

@patch("app.api.routes.ask.rag_service")
def test_ask_endpoint_default_top_k(mock_rag):
    mock_rag.answer.return_value = AskResponse(
        query="How to create a POST endpoint?",
        answer="Use `@app.post()`.",
        sources=[],
        total_sources=0,
        model="llama-3.3-70b-versatile"
    )

    resp = client.post("/api/v1/ask/", json={"query": "How to create a POST endpoint?"})
    assert resp.status_code == 200
    mock_rag.answer.assert_called_once_with(query="How to create a POST endpoint?", top_k=8)

def test_ask_endpoint_empty_query():
    resp = client.post("/api/v1/ask/", json={"query": ""})
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"].lower()

def test_ask_endpoint_whitespace_query():
    resp = client.post("/api/v1/ask/", json={"query": "   \n\t  "})
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"].lower()

def test_ask_endpoint_invalid_top_k():
    resp = client.post("/api/v1/ask/", json={"query": "valid query", "top_k": 0})
    assert resp.status_code == 422

    resp = client.post("/api/v1/ask/", json={"query": "valid query", "top_k": -5})
    assert resp.status_code == 422

    resp = client.post("/api/v1/ask/", json={"query": "valid query", "top_k": 100})
    assert resp.status_code == 422

@patch("app.api.routes.ask.rag_service")
def test_ask_endpoint_missing_groq_config(mock_rag):
    mock_rag.answer.side_effect = GroqConfigurationError("GROQ_API_KEY is not configured.")

    resp = client.post("/api/v1/ask/", json={"query": "How to create a POST endpoint?"})
    assert resp.status_code == 503
    assert "LLM service is not configured" in resp.json()["detail"]

@patch("app.api.routes.ask.rag_service")
def test_ask_endpoint_missing_vector_index(mock_rag):
    mock_rag.answer.side_effect = IndexUnavailableError("Index file not found.")

    resp = client.post("/api/v1/ask/", json={"query": "How to create a POST endpoint?"})
    assert resp.status_code == 503
    assert "Search index unavailable" in resp.json()["detail"]

@patch("app.api.routes.ask.rag_service")
def test_ask_endpoint_groq_timeout(mock_rag):
    mock_rag.answer.side_effect = GroqTimeoutError("Timed out.")

    resp = client.post("/api/v1/ask/", json={"query": "How to create a POST endpoint?"})
    assert resp.status_code == 504
    assert "timed out" in resp.json()["detail"].lower()

@patch("app.api.routes.ask.rag_service")
def test_ask_endpoint_groq_provider_error(mock_rag):
    mock_rag.answer.side_effect = GroqProviderError("Rate limit reached")

    resp = client.post("/api/v1/ask/", json={"query": "How to create a POST endpoint?"})
    assert resp.status_code == 502
    assert "Upstream LLM provider error" in resp.json()["detail"]

@patch("app.api.routes.ask.rag_service")
def test_ask_endpoint_internal_error(mock_rag):
    mock_rag.answer.side_effect = RuntimeError("Crash")

    resp = client.post("/api/v1/ask/", json={"query": "How to create a POST endpoint?"})
    assert resp.status_code == 500
    assert "unexpected internal server error" in resp.json()["detail"].lower()

def test_retrieve_endpoint_still_works():
    # Ensure POST /api/v1/retrieve/ is preserved and functional
    resp = client.post("/api/v1/retrieve/", json={"query": ""})
    assert resp.status_code == 422
