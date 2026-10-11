import pytest
from unittest.mock import MagicMock
from app.retrieval.models import RetrievalResult
from app.rag.service import RAGService
from app.rag.context import build_context
from app.rag.models import SourceCitation
from app.llm.client import GroqLLMClient, GroqProviderError

@pytest.fixture
def sample_retrieval_results():
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            text="Use @app.post('/items') to define a POST endpoint in FastAPI.",
            source_url="https://fastapi.tiangolo.com/tutorial/first-steps/",
            document_id="doc_fastapi",
            section_title="First Steps",
            score=0.88
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            text="Request bodies are defined using Pydantic models with BaseModel.",
            source_url="https://fastapi.tiangolo.com/tutorial/body/",
            document_id="doc_fastapi",
            section_title="Request Body",
            score=0.82
        )
    ]

def test_build_context_structure(sample_retrieval_results):
    context, citations = build_context(sample_retrieval_results, max_chars=4000)
    assert len(citations) == 2
    assert citations[0].source_id == "S1"
    assert citations[0].chunk_id == "chunk_1"
    assert citations[1].source_id == "S2"
    assert "--- Source [S1] ---" in context
    assert "--- Source [S2] ---" in context
    assert "Use @app.post" in context

def test_build_context_budget_bounding(sample_retrieval_results):
    # Very small budget that only fits the first chunk
    context, citations = build_context(sample_retrieval_results, max_chars=120)
    assert len(citations) == 1
    assert citations[0].source_id == "S1"
    assert "[S2]" not in context

def test_rag_service_success(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "llama-3.3-70b-versatile"
    mock_llm.generate.return_value = "Define a POST endpoint using `@app.post()` [S1] and request bodies [S2]."

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp = service.answer("How to create a POST endpoint?", top_k=2)

    assert resp.query == "How to create a POST endpoint?"
    assert "Define a POST endpoint" in resp.answer
    assert resp.total_sources == 2
    assert resp.sources[0].source_id == "S1"
    assert str(resp.sources[0].source_url) == "https://fastapi.tiangolo.com/tutorial/first-steps/"
    assert resp.model == "llama-3.3-70b-versatile"
    mock_retrieval.retrieve.assert_called_once_with("How to create a POST endpoint?", top_k=2)
    mock_llm.generate.assert_called_once()

def test_rag_service_empty_retrieval():
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = []

    mock_llm = MagicMock()
    mock_llm.model = "llama-3.3-70b-versatile"

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp = service.answer("Quantum entanglement in FastAPI?", top_k=5)

    assert resp.total_sources == 0
    assert resp.sources == []
    assert "not contain enough relevant information" in resp.answer
    # LLM must NOT be called when retrieval returns 0 chunks
    mock_llm.generate.assert_not_called()

def test_rag_service_empty_or_whitespace_query():
    service = RAGService()
    with pytest.raises(ValueError, match="empty or whitespace only"):
        service.answer("")

    with pytest.raises(ValueError, match="empty or whitespace only"):
        service.answer("   \n\t  ")

def test_rag_service_invalid_top_k():
    service = RAGService()
    with pytest.raises(ValueError, match="positive integer"):
        service.answer("valid query", top_k=0)

    with pytest.raises(ValueError, match="positive integer"):
        service.answer("valid query", top_k=-3)

def test_rag_service_propagates_provider_error(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.generate.side_effect = GroqProviderError("Groq service unavailable")

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    with pytest.raises(GroqProviderError, match="Groq service unavailable"):
        service.answer("query", top_k=2)
