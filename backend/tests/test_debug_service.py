import json
import pytest
from unittest.mock import MagicMock
from app.retrieval.models import RetrievalResult
from app.debug.models import DebugRequest, LikelyCause, SuggestedFix
from app.debug.service import (
    DebugService,
    sanitize_diagnostic_input,
    build_debug_retrieval_query,
    parse_debug_json,
)
from app.llm.client import GroqProviderError, GroqTimeoutError, GroqConfigurationError

@pytest.fixture
def sample_retrieval_results():
    return [
        RetrievalResult(
            chunk_id="chunk_1",
            text="FastAPI raises 422 Unprocessable Entity when request validation fails on Pydantic models.",
            source_url="https://fastapi.tiangolo.com/tutorial/body/",
            document_id="doc_fastapi_body",
            section_title="Request Body",
            score=0.89
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            text="Use Header() to declare header parameters in path operation functions.",
            source_url="https://fastapi.tiangolo.com/tutorial/header-params/",
            document_id="doc_fastapi_header",
            section_title="Header Parameters",
            score=0.78
        )
    ]

def test_sanitize_diagnostic_input_redaction():
    raw = (
        "Exception in request: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz123\n"
        "api_key = 'sk-proj-1234567890abcdef'\n"
        "DATABASE_URL: postgresql://admin:supersecretpassword@localhost:5432/devdocs"
    )
    sanitized = sanitize_diagnostic_input(raw)
    assert "Bearer [REDACTED]" in sanitized
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized
    assert "sk-proj-1234567890abcdef" not in sanitized
    assert "supersecretpassword" not in sanitized
    assert "[REDACTED]" in sanitized

def test_build_debug_retrieval_query():
    error_trace = (
        'Traceback (most recent call last):\n'
        '  File "/app/main.py", line 42, in endpoint\n'
        'pydantic.error_wrappers.ValidationError: 1 validation error for Item\n'
        'price\n'
        '  field required (type=value_error.missing)'
    )
    query = build_debug_retrieval_query(
        error_message=error_trace,
        framework="FastAPI",
        language="Python",
        context="POST request to /items endpoint"
    )
    assert "FastAPI" in query
    assert "Python" in query
    assert "ValidationError" in query or "field required" in query
    assert len(query) <= 250

def test_parse_debug_json_fences():
    raw_md = '```json\n{"error_summary": "Test summary", "likely_causes": []}\n```'
    parsed = parse_debug_json(raw_md)
    assert parsed["error_summary"] == "Test summary"

def test_debug_service_success(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"
    mock_llm.generate.return_value = json.dumps({
        "error_summary": "Pydantic validation failure on request payload",
        "likely_causes": [
            {
                "cause": "Missing required field",
                "explanation": "FastAPI returns 422 if required body attributes are omitted [S1].",
                "citations": ["S1"]
            }
        ],
        "diagnostic_checks": [
            "Check that the client JSON body contains the price field",
            "Verify Pydantic model field definitions"
        ],
        "suggested_fixes": [
            {
                "fix": "Provide default value or Optional type",
                "explanation": "Declare the field as Optional[float] = None if it is not mandatory [S1].",
                "citations": ["S1"]
            }
        ],
        "uncertainty_notes": [
            "The client request body content was not provided in the error message."
        ]
    })

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    req = DebugRequest(
        error_message="ValidationError: field required (type=value_error.missing)",
        framework="FastAPI",
        language="Python"
    )
    resp = service.diagnose(req)

    assert "Pydantic validation failure" in resp.error_summary
    assert len(resp.likely_causes) == 1
    assert resp.likely_causes[0].cause == "Missing required field"
    assert "S1" in resp.likely_causes[0].citations
    assert len(resp.suggested_fixes) == 1
    assert resp.suggested_fixes[0].citations == ["S1"]
    assert len(resp.diagnostic_checks) == 2
    assert len(resp.uncertainty_notes) == 1
    # Only S1 cited, so S2 should NOT be in sources
    assert resp.total_sources == 1
    assert len(resp.sources) == 1
    assert resp.sources[0].source_id == "S1"
    assert resp.model == "openai/gpt-oss-120b"
    mock_llm.generate.assert_called_once()

def test_debug_service_empty_retrieval_no_llm_call():
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = []

    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    req = DebugRequest(
        error_message="java.lang.NullPointerException at com.example.service.UserService.getUser",
        framework="Spring Boot",
        language="Java"
    )
    resp = service.diagnose(req)

    assert resp.total_sources == 0
    assert resp.sources == []
    assert "does not match any indexed documentation" in resp.error_summary
    assert len(resp.likely_causes) >= 1
    assert "Insufficient Documentation Evidence" in resp.likely_causes[0].cause
    # Verify LLM was NOT called
    mock_llm.generate.assert_not_called()

def test_debug_service_unknown_citation_corrective_retry_success(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"

    # First attempt cites unauthorized [S9]
    attempt1 = json.dumps({
        "error_summary": "Bad input",
        "likely_causes": [{"cause": "Wrong type", "explanation": "Detail", "citations": ["S9"]}],
        "diagnostic_checks": ["Check types"],
        "suggested_fixes": [{"fix": "Fix type", "explanation": "Detail", "citations": ["S9"]}],
        "uncertainty_notes": []
    })
    # Second attempt fixes citation to valid [S1]
    attempt2 = json.dumps({
        "error_summary": "Bad input",
        "likely_causes": [{"cause": "Wrong type", "explanation": "Detail", "citations": ["S1"]}],
        "diagnostic_checks": ["Check types"],
        "suggested_fixes": [{"fix": "Fix type", "explanation": "Detail", "citations": ["S1"]}],
        "uncertainty_notes": []
    })
    mock_llm.generate.side_effect = [attempt1, attempt2]

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    req = DebugRequest(error_message="ValidationError: input error")
    resp = service.diagnose(req)

    assert mock_llm.generate.call_count == 2
    assert resp.total_sources == 1
    assert resp.sources[0].source_id == "S1"
    assert resp.likely_causes[0].citations == ["S1"]

def test_debug_service_unknown_citation_retry_exhausted_fallback(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"

    bad_json = json.dumps({
        "error_summary": "Bad input",
        "likely_causes": [{"cause": "Bad type", "explanation": "Detail", "citations": ["S9"]}],
        "diagnostic_checks": ["Check types"],
        "suggested_fixes": [{"fix": "Fix type", "explanation": "Detail", "citations": ["S9"]}],
        "uncertainty_notes": []
    })
    mock_llm.generate.side_effect = [bad_json, bad_json]

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    req = DebugRequest(error_message="ValidationError: input error")
    resp = service.diagnose(req)

    assert mock_llm.generate.call_count == 2
    assert resp.total_sources == 0
    assert resp.sources == []
    assert "Insufficient documentation evidence" in resp.error_summary

def test_debug_service_fabricated_url_retry_and_fallback(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"

    fabricated_json = json.dumps({
        "error_summary": "Bad input",
        "likely_causes": [{
            "cause": "Bad input",
            "explanation": "See https://fastapi.tiangolo.com/hallucinated-guide/ [S1]",
            "citations": ["S1"]
        }],
        "diagnostic_checks": ["Check types"],
        "suggested_fixes": [],
        "uncertainty_notes": []
    })
    mock_llm.generate.side_effect = [fabricated_json, fabricated_json]

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    req = DebugRequest(error_message="ValidationError: input error")
    resp = service.diagnose(req)

    assert mock_llm.generate.call_count == 2
    assert resp.total_sources == 0
    assert resp.sources == []
    assert "Insufficient documentation evidence" in resp.error_summary

def test_debug_service_malformed_json_retry_success(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"

    valid_json = json.dumps({
        "error_summary": "Resolved after fix",
        "likely_causes": [{"cause": "Known issue", "explanation": "Explanation [S1]", "citations": ["S1"]}],
        "diagnostic_checks": ["Check 1"],
        "suggested_fixes": [{"fix": "Fix 1", "explanation": "Fix explanation [S1]", "citations": ["S1"]}],
        "uncertainty_notes": []
    })
    mock_llm.generate.side_effect = ["This is not valid JSON at all!", valid_json]

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    req = DebugRequest(error_message="SyntaxError in main.py")
    resp = service.diagnose(req)

    assert mock_llm.generate.call_count == 2
    assert resp.error_summary == "Resolved after fix"
    assert resp.total_sources == 1

def test_debug_service_propagates_groq_exceptions(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.generate.side_effect = GroqProviderError("Groq 500 error")

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    req = DebugRequest(error_message="ValidationError")

    with pytest.raises(GroqProviderError, match="Groq 500 error"):
        service.diagnose(req)

def test_debug_service_request_isolation():
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
    json_A = json.dumps({
        "error_summary": "Error A",
        "likely_causes": [{"cause": "Cause A", "explanation": "Exp [S1]", "citations": ["S1"]}],
        "diagnostic_checks": [],
        "suggested_fixes": [],
        "uncertainty_notes": []
    })
    json_B = json.dumps({
        "error_summary": "Error B",
        "likely_causes": [{"cause": "Cause B", "explanation": "Exp [S1]", "citations": ["S1"]}],
        "diagnostic_checks": [],
        "suggested_fixes": [],
        "uncertainty_notes": []
    })
    mock_llm.generate.side_effect = [json_A, json_B]

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    resp1 = service.diagnose(DebugRequest(error_message="Error A"))
    resp2 = service.diagnose(DebugRequest(error_message="Error B"))

    assert resp1.sources[0].document_id == "doc_A"
    assert resp2.sources[0].document_id == "doc_B"
    assert resp1.sources[0].source_url != resp2.sources[0].source_url

def test_debug_service_schema_validation_retry_and_success(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"

    # Attempt 1 has likely_causes as strings instead of dicts (fails Pydantic schema validation)
    bad_schema = json.dumps({
        "error_summary": "Summary",
        "likely_causes": ["This is an invalid string cause"],
        "diagnostic_checks": ["Check 1"],
        "suggested_fixes": [],
        "uncertainty_notes": []
    })
    # Attempt 2 has valid schema
    good_schema = json.dumps({
        "error_summary": "Valid summary",
        "likely_causes": [{"cause": "Valid cause", "explanation": "Valid explanation [S1]", "citations": ["S1"]}],
        "diagnostic_checks": ["Check 1"],
        "suggested_fixes": [{"fix": "Fix", "explanation": "Fix explanation [S1]", "citations": ["S1"]}],
        "uncertainty_notes": []
    })
    mock_llm.generate.side_effect = [bad_schema, good_schema]

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    req = DebugRequest(error_message="Error in app")
    resp = service.diagnose(req)

    assert mock_llm.generate.call_count == 2
    assert resp.error_summary == "Valid summary"
    assert resp.total_sources == 1

def test_debug_service_schema_validation_retry_and_fallback(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"

    # Both attempts fail schema validation
    bad_schema = json.dumps({
        "error_summary": "Summary",
        "likely_causes": ["Still an invalid string cause"],
        "diagnostic_checks": ["Check 1"]
    })
    mock_llm.generate.side_effect = [bad_schema, bad_schema]

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    req = DebugRequest(error_message="Error in app")
    resp = service.diagnose(req)

    assert mock_llm.generate.call_count == 2
    assert resp.total_sources == 0
    assert "Insufficient documentation evidence" in resp.error_summary

def test_debug_service_lookalike_url_retry_and_fallback(sample_retrieval_results):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = sample_retrieval_results

    mock_llm = MagicMock()
    mock_llm.model = "openai/gpt-oss-120b"

    lookalike_json = json.dumps({
        "error_summary": "Summary",
        "likely_causes": [{
            "cause": "Lookalike URL attempt",
            "explanation": "See https://fastapi.tiangolo.com.evil.com/fake-page/ [S1]",
            "citations": ["S1"]
        }],
        "diagnostic_checks": ["Check types"],
        "suggested_fixes": [],
        "uncertainty_notes": []
    })
    mock_llm.generate.side_effect = [lookalike_json, lookalike_json]

    service = DebugService(retrieval_svc=mock_retrieval, llm=mock_llm)
    req = DebugRequest(error_message="ValidationError: input error")
    resp = service.diagnose(req)

    assert mock_llm.generate.call_count == 2
    assert resp.total_sources == 0
    assert resp.sources == []
    assert "Insufficient documentation evidence" in resp.error_summary
