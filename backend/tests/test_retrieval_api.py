import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.retrieval.service import IndexUnavailableError
from app.retrieval.models import RetrievalResult

client = TestClient(app)

@patch("app.api.routes.retrieve.retrieval_service")
def test_retrieve_endpoint_success(mock_service):
    mock_service.retrieve.return_value = [
        RetrievalResult(
            chunk_id="c1",
            text="text 1",
            source_url="http://a.com",
            document_id="d1",
            section_title="sec 1",
            score=0.9
        )
    ]
    
    response = client.post("/api/v1/retrieve/", json={"query": "test query", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert data["total_results"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["chunk_id"] == "c1"

@patch("app.api.routes.retrieve.retrieval_service")
def test_retrieve_endpoint_default_top_k(mock_service):
    mock_service.retrieve.return_value = []
    
    response = client.post("/api/v1/retrieve/", json={"query": "test query"})
    assert response.status_code == 200
    mock_service.retrieve.assert_called_once_with("test query", 8)

@patch("app.api.routes.retrieve.retrieval_service")
def test_retrieve_endpoint_empty_query(mock_service):
    response = client.post("/api/v1/retrieve/", json={"query": "   ", "top_k": 3})
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()

@patch("app.api.routes.retrieve.retrieval_service")
def test_retrieve_endpoint_invalid_top_k(mock_service):
    response = client.post("/api/v1/retrieve/", json={"query": "test", "top_k": 0})
    assert response.status_code == 422
    
    response = client.post("/api/v1/retrieve/", json={"query": "test", "top_k": -5})
    assert response.status_code == 422
    
    response = client.post("/api/v1/retrieve/", json={"query": "test", "top_k": "invalid"})
    assert response.status_code == 422

@patch("app.api.routes.retrieve.retrieval_service")
def test_retrieve_endpoint_index_unavailable(mock_service):
    mock_service.retrieve.side_effect = IndexUnavailableError("Index not found.")
    
    response = client.post("/api/v1/retrieve/", json={"query": "test"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Index not found."

@patch("app.api.routes.retrieve.retrieval_service")
def test_retrieve_endpoint_internal_error(mock_service):
    mock_service.retrieve.side_effect = RuntimeError("Something bad happened.")
    
    response = client.post("/api/v1/retrieve/", json={"query": "test"})
    assert response.status_code == 500
    assert response.json()["detail"] == "An internal server error occurred."
