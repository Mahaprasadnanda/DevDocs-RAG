import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.debug.models import DebugResponse, LikelyCause, SuggestedFix
from app.rag.models import SourceCitation
from app.retrieval.service import IndexUnavailableError
from app.llm.client import GroqConfigurationError, GroqTimeoutError, GroqProviderError

client = TestClient(app)

@patch("app.api.routes.debug.debug_service")
def test_debug_endpoint_success(mock_debug):
    mock_debug.diagnose.return_value = DebugResponse(
        error_summary="Pydantic validation error in request body",
        likely_causes=[
            LikelyCause(
                cause="Missing required field",
                explanation="Field 'name' is required by Pydantic model [S1].",
                citations=["S1"]
            )
        ],
        diagnostic_checks=["Check client request body JSON payload"],
        suggested_fixes=[
            SuggestedFix(
                fix="Supply required field in payload",
                explanation="Ensure client sends 'name' in JSON object [S1].",
                citations=["S1"]
            )
        ],
        sources=[
            SourceCitation(
                source_id="S1",
                chunk_id="chunk_101",
                document_id="doc_fastapi_body",
                section_title="Request Body",
                source_url="https://fastapi.tiangolo.com/tutorial/body/",
                score=0.88
            )
        ],
        total_sources=1,
        uncertainty_notes=[],
        model="openai/gpt-oss-120b"
    )

    payload = {
        "error_message": "ValidationError: 1 validation error for Item\nname\n  field required",
        "framework": "FastAPI",
        "language": "Python",
        "context": "Calling POST /items",
        "top_k": 5,
        "min_score": 0.35
    }
    resp = client.post("/api/v1/debug/", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["error_summary"] == "Pydantic validation error in request body"
    assert len(data["likely_causes"]) == 1
    assert data["likely_causes"][0]["cause"] == "Missing required field"
    assert data["likely_causes"][0]["citations"] == ["S1"]
    assert len(data["suggested_fixes"]) == 1
    assert data["suggested_fixes"][0]["citations"] == ["S1"]
    assert data["total_sources"] == 1
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source_id"] == "S1"
    assert data["model"] == "openai/gpt-oss-120b"

@patch("app.api.routes.debug.debug_service")
def test_debug_endpoint_without_trailing_slash(mock_debug):
    mock_debug.diagnose.return_value = DebugResponse(
        error_summary="Summary",
        likely_causes=[],
        diagnostic_checks=[],
        suggested_fixes=[],
        sources=[],
        total_sources=0,
        uncertainty_notes=[],
        model="openai/gpt-oss-120b"
    )

    resp = client.post("/api/v1/debug", json={"error_message": "Some error"})
    assert resp.status_code == 200

def test_debug_endpoint_empty_error_message():
    resp = client.post("/api/v1/debug/", json={"error_message": ""})
    assert resp.status_code == 422
    detail = str(resp.json()["detail"]).lower()
    assert "at least 1 character" in detail or "empty" in detail

def test_debug_endpoint_whitespace_error_message():
    resp = client.post("/api/v1/debug/", json={"error_message": "   \n\t  "})
    assert resp.status_code == 422
    assert "empty" in str(resp.json()["detail"]).lower()

def test_debug_endpoint_missing_error_message():
    resp = client.post("/api/v1/debug/", json={"framework": "FastAPI"})
    assert resp.status_code == 422

def test_debug_endpoint_invalid_top_k():
    resp = client.post("/api/v1/debug/", json={"error_message": "Error", "top_k": 0})
    assert resp.status_code == 422

    resp = client.post("/api/v1/debug/", json={"error_message": "Error", "top_k": 25})
    assert resp.status_code == 422

def test_debug_endpoint_invalid_min_score():
    resp = client.post("/api/v1/debug/", json={"error_message": "Error", "min_score": -0.1})
    assert resp.status_code == 422

    resp = client.post("/api/v1/debug/", json={"error_message": "Error", "min_score": 1.5})
    assert resp.status_code == 422

def test_debug_endpoint_error_message_too_long():
    resp = client.post("/api/v1/debug/", json={"error_message": "x" * 10001})
    assert resp.status_code == 422

@patch("app.api.routes.debug.debug_service")
def test_debug_endpoint_missing_groq_config(mock_debug):
    mock_debug.diagnose.side_effect = GroqConfigurationError("GROQ_API_KEY is not configured.")

    resp = client.post("/api/v1/debug/", json={"error_message": "RuntimeError: Something broke"})
    assert resp.status_code == 503
    assert "LLM service is not configured" in resp.json()["detail"]

@patch("app.api.routes.debug.debug_service")
def test_debug_endpoint_missing_vector_index(mock_debug):
    mock_debug.diagnose.side_effect = IndexUnavailableError("FAISS index not found.")

    resp = client.post("/api/v1/debug/", json={"error_message": "RuntimeError: Something broke"})
    assert resp.status_code == 503
    assert "Search index unavailable" in resp.json()["detail"]

@patch("app.api.routes.debug.debug_service")
def test_debug_endpoint_groq_timeout(mock_debug):
    mock_debug.diagnose.side_effect = GroqTimeoutError("Timed out")

    resp = client.post("/api/v1/debug/", json={"error_message": "RuntimeError: Something broke"})
    assert resp.status_code == 504
    assert "timed out" in resp.json()["detail"].lower()

@patch("app.api.routes.debug.debug_service")
def test_debug_endpoint_groq_provider_error(mock_debug):
    mock_debug.diagnose.side_effect = GroqProviderError("Bad Gateway from upstream")

    resp = client.post("/api/v1/debug/", json={"error_message": "RuntimeError: Something broke"})
    assert resp.status_code == 502
    assert "Upstream LLM provider error" in resp.json()["detail"]

@patch("app.api.routes.debug.debug_service")
def test_debug_endpoint_internal_error(mock_debug):
    mock_debug.diagnose.side_effect = RuntimeError("Fatal crash")

    resp = client.post("/api/v1/debug/", json={"error_message": "RuntimeError: Something broke"})
    assert resp.status_code == 500
    assert "unexpected internal server error" in resp.json()["detail"].lower()

def test_existing_endpoints_preserved():
    # Verify health, retrieve, and ask endpoints are untouched
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200

    retrieve_resp = client.post("/api/v1/retrieve/", json={"query": ""})
    assert retrieve_resp.status_code == 422

    ask_resp = client.post("/api/v1/ask/", json={"query": ""})
    assert ask_resp.status_code == 422
