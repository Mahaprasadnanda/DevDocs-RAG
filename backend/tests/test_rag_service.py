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
    mock_retrieval.retrieve.assert_called_once_with("How to create a POST endpoint?", top_k=2, min_score=None)
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

def test_rag_service_out_of_domain_no_sources():
    # Verify that an out-of-domain query (e.g. TCP vs UDP) which yields [] from retrieval
    # results in an unanswerable response with zero citations and no LLM hallucination
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = []

    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp = service.answer("What is the difference between TCP and UDP?", top_k=5)

    assert resp.total_sources == 0
    assert resp.sources == []
    assert "not contain enough relevant information" in resp.answer
    mock_llm.generate.assert_not_called()

def test_rag_service_passes_min_score():
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = []
    mock_llm = MagicMock()
    mock_llm.model = "test-model"

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    service.answer("query", top_k=4, min_score=0.45)
    mock_retrieval.retrieve.assert_called_once_with("query", top_k=4, min_score=0.45)

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

def test_rag_service_subset_sources_returned(sample_retrieval_results):
    # If 2 chunks retrieved, but answer only cites [S1], only S1 should be in returned sources
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.generate.return_value = "Only citing the first source here [S1]."

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp = service.answer("query", top_k=2)

    assert resp.total_sources == 1
    assert len(resp.sources) == 1
    assert resp.sources[0].source_id == "S1"

def test_rag_service_unknown_citation_corrective_retry_success(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    # First attempt returns invalid citation [S9], second attempt corrects to [S1]
    mock_llm.generate.side_effect = [
        "First attempt references unauthorized [S9].",
        "Corrected response references valid source [S1]."
    ]

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp = service.answer("query", top_k=2)

    assert mock_llm.generate.call_count == 2
    assert "Corrected response references valid source [S1]" in resp.answer
    assert resp.total_sources == 1
    assert resp.sources[0].source_id == "S1"

def test_rag_service_unknown_citation_retry_exhausted_fallback(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    # Both attempts fail with unauthorized citation [S9]
    mock_llm.generate.side_effect = [
        "First attempt references unauthorized [S9].",
        "Second attempt still references unauthorized [S9]."
    ]

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp = service.answer("query", top_k=2)

    assert mock_llm.generate.call_count == 2
    assert resp.total_sources == 0
    assert resp.sources == []
    assert "not contain enough relevant information" in resp.answer

def test_rag_service_fabricated_url_retry_and_fallback(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    # Both attempts include fabricated URL
    mock_llm.generate.side_effect = [
        "Check out https://fastapi.tiangolo.com/nonexistent-feature/ [S1].",
        "Still check https://fastapi.tiangolo.com/nonexistent-feature/ [S1]."
    ]

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp = service.answer("query", top_k=2)

    assert resp.total_sources == 0
    assert resp.sources == []
    assert "not contain enough relevant information" in resp.answer

def test_rag_service_missing_citations_retry_success(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    # Attempt 1 has no citations, attempt 2 provides [S1] and [S2]
    mock_llm.generate.side_effect = [
        "FastAPI is an async framework without citations.",
        "FastAPI is an async framework [S1] supporting request bodies [S2]."
    ]

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp = service.answer("query", top_k=2)

    assert mock_llm.generate.call_count == 2
    assert resp.total_sources == 2
    assert "[S1]" in resp.answer

def test_rag_service_repeated_citations_preserves_unique_sources(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.generate.return_value = "Use @app.get() [S1]. Also [S1] is fast. Body in [S2]."

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp = service.answer("query", top_k=2)

    assert resp.total_sources == 2
    assert [s.source_id for s in resp.sources] == ["S1", "S2"]

def test_rag_service_stateless_no_leak_between_requests():
    # Request 1 retrieves doc_A with S1
    # Request 2 retrieves doc_B with S1
    res_A = [
        RetrievalResult(
            chunk_id="chunk_A",
            text="Text A",
            source_url="https://fastapi.tiangolo.com/docA/",
            document_id="doc_A",
            section_title="Doc A",
            score=0.9
        )
    ]
    res_B = [
        RetrievalResult(
            chunk_id="chunk_B",
            text="Text B",
            source_url="https://fastapi.tiangolo.com/docB/",
            document_id="doc_B",
            section_title="Doc B",
            score=0.9
        )
    ]

    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.side_effect = [res_A, res_B]

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.generate.side_effect = [
        "Answer A citing [S1].",
        "Answer B citing [S1]."
    ]

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp1 = service.answer("query A")
    resp2 = service.answer("query B")

    assert resp1.sources[0].document_id == "doc_A"
    assert resp2.sources[0].document_id == "doc_B"
    assert resp1.sources[0].source_url != resp2.sources[0].source_url

def test_rag_service_explicit_refusal_empty_sources(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.generate.return_value = "The provided documentation does not contain enough information to answer this question."

    service = RAGService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp = service.answer("How to build rockets in FastAPI?")

    assert resp.total_sources == 0
    assert resp.sources == []
    assert "not contain enough information" in resp.answer
