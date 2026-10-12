import pytest
from app.rag.models import SourceCitation
from app.rag.validator import (
    validate_citations,
    extract_citations,
    extract_external_urls,
    is_refusal_response,
    normalize_citation_syntax,
)

@pytest.fixture
def mock_citations():
    return [
        SourceCitation(
            source_id="S1",
            chunk_id="chunk_1",
            document_id="doc_1",
            section_title="First Steps",
            source_url="https://fastapi.tiangolo.com/tutorial/first-steps/",
            score=0.89
        ),
        SourceCitation(
            source_id="S2",
            chunk_id="chunk_2",
            document_id="doc_2",
            section_title="Path Parameters",
            source_url="https://fastapi.tiangolo.com/tutorial/path-params/",
            score=0.82
        )
    ]

def test_validator_valid_citations(mock_citations):
    text = "FastAPI endpoints are defined using `@app.get()` [S1] and path parameters [S2]."
    result = validate_citations(text, mock_citations)
    assert result.is_valid is True
    assert result.is_refusal is False
    assert result.cited_source_ids == ["S1", "S2"]
    assert result.invalid_source_ids == []
    assert result.fabricated_urls == []
    assert result.error_message is None

def test_validator_unknown_citation_identifier(mock_citations):
    # Cites S3 which is not in the allowed S1, S2 set
    text = "FastAPI supports background tasks [S3]."
    result = validate_citations(text, mock_citations)
    assert result.is_valid is False
    assert "S3" in result.invalid_source_ids
    assert "unauthorized source identifier" in result.error_message.lower()

def test_validator_fabricated_documentation_url(mock_citations):
    text = "See details at https://fastapi.tiangolo.com/nonexistent-feature/guide/ [S1]."
    result = validate_citations(text, mock_citations)
    assert result.is_valid is False
    assert len(result.fabricated_urls) == 1
    assert "fabricated documentation url" in result.error_message.lower()

def test_validator_allowed_url_subpath_or_anchor(mock_citations):
    text = "Refer to https://fastapi.tiangolo.com/tutorial/path-params/#order-matters [S2]."
    result = validate_citations(text, mock_citations)
    assert result.is_valid is True
    assert result.fabricated_urls == []

def test_validator_local_example_url_ignored(mock_citations):
    text = "Start the app and visit `http://127.0.0.1:8000/items/5` or `http://localhost:8000` [S1]."
    result = validate_citations(text, mock_citations)
    assert result.is_valid is True
    assert result.fabricated_urls == []

def test_validator_no_citations_on_informative_answer(mock_citations):
    text = "FastAPI uses Starlette and Pydantic for high performance async endpoints."
    result = validate_citations(text, mock_citations)
    assert result.is_valid is False
    assert "no inline citations" in result.error_message.lower()

def test_validator_refusal_accepted_without_citations(mock_citations):
    refusals = [
        "The provided documentation does not contain enough information to answer this question.",
        "The indexed documentation does not contain enough relevant information to answer this question.",
        "This question cannot be answered based on the provided documentation."
    ]
    for text in refusals:
        result = validate_citations(text, mock_citations)
        assert result.is_valid is True
        assert result.is_refusal is True
        assert result.cited_source_ids == []

def test_validator_repeated_citations_deduplicated(mock_citations):
    text = "Use @app.get() [S1]. As noted in [S1], and with body [S2]."
    result = validate_citations(text, mock_citations)
    assert result.is_valid is True
    assert result.cited_source_ids == ["S1", "S2"]

def test_validator_case_normalization(mock_citations):
    text = "Define an endpoint using `@app.get()` [s1] and path params [S2]."
    result = validate_citations(text, mock_citations)
    assert result.is_valid is True
    assert result.cited_source_ids == ["S1", "S2"]
    assert "[S1]" in result.sanitized_answer

def test_validator_malformed_citation_syntax(mock_citations):
    text = "Refer to [Source 1] and [Doc 2]."
    result = validate_citations(text, mock_citations)
    assert result.is_valid is False

def test_validator_empty_or_whitespace_answer(mock_citations):
    result = validate_citations("   \n\t ", mock_citations)
    assert result.is_valid is False
    assert "empty" in result.error_message.lower()

# --- Milestone 9 Hardening Tests: URL Boundaries, Lookalike Domains, All-Field Validation ---

from app.rag.validator import validate_debug_response_citations, is_url_allowed
from app.debug.models import RawDebugPayload
from pydantic import ValidationError

def test_is_url_allowed_rejects_lookalike_and_deceptive_domains(mock_citations):
    allowed_urls = {str(c.source_url).rstrip("/") for c in mock_citations}
    # Lookalike suffix domain
    assert is_url_allowed("https://fastapi.tiangolo.com.evil.com/tutorial/first-steps/", allowed_urls) is False
    # Userinfo deceptive host trick
    assert is_url_allowed("https://fastapi.tiangolo.com@evil.com/tutorial/first-steps/", allowed_urls) is False
    # Scheme mismatch
    assert is_url_allowed("http://fastapi.tiangolo.com/tutorial/first-steps/", allowed_urls) is False
    # Port mismatch
    assert is_url_allowed("https://fastapi.tiangolo.com:8443/tutorial/first-steps/", allowed_urls) is False
    # Unindexed path
    assert is_url_allowed("https://fastapi.tiangolo.com/unindexed-section/", allowed_urls) is False
    # Exact allowed path
    assert is_url_allowed("https://fastapi.tiangolo.com/tutorial/first-steps/", allowed_urls) is True
    # Allowed subpath
    assert is_url_allowed("https://fastapi.tiangolo.com/tutorial/first-steps/advanced/", allowed_urls) is True
    # Allowed fragment
    assert is_url_allowed("https://fastapi.tiangolo.com/tutorial/first-steps/#summary", allowed_urls) is True

def test_debug_validator_unauthorized_citation_in_error_summary(mock_citations):
    debug_dict = {
        "error_summary": "Error occurring in validation [S99]",
        "likely_causes": [{"cause": "Known cause", "explanation": "See [S1]", "citations": ["S1"]}],
        "diagnostic_checks": ["Check code"],
        "suggested_fixes": [{"fix": "Fix code", "explanation": "See [S1]", "citations": ["S1"]}],
        "uncertainty_notes": []
    }
    res = validate_debug_response_citations(debug_dict, str(debug_dict), mock_citations)
    assert res.is_valid is False
    assert "S99" in res.invalid_source_ids

def test_debug_validator_unauthorized_citation_in_cause_title(mock_citations):
    debug_dict = {
        "error_summary": "Error summary",
        "likely_causes": [{"cause": "Known cause [S99]", "explanation": "See [S1]", "citations": ["S1"]}],
        "diagnostic_checks": ["Check code"],
        "suggested_fixes": [],
        "uncertainty_notes": []
    }
    res = validate_debug_response_citations(debug_dict, str(debug_dict), mock_citations)
    assert res.is_valid is False
    assert "S99" in res.invalid_source_ids

def test_debug_validator_unauthorized_citation_in_diagnostic_checks(mock_citations):
    debug_dict = {
        "error_summary": "Error summary",
        "likely_causes": [{"cause": "Known cause", "explanation": "See [S1]", "citations": ["S1"]}],
        "diagnostic_checks": ["Check model definition [S99]"],
        "suggested_fixes": [],
        "uncertainty_notes": []
    }
    res = validate_debug_response_citations(debug_dict, str(debug_dict), mock_citations)
    assert res.is_valid is False
    assert "S99" in res.invalid_source_ids

def test_debug_validator_unauthorized_citation_in_suggested_fix_title(mock_citations):
    debug_dict = {
        "error_summary": "Error summary",
        "likely_causes": [{"cause": "Known cause", "explanation": "See [S1]", "citations": ["S1"]}],
        "diagnostic_checks": [],
        "suggested_fixes": [{"fix": "Update model [S99]", "explanation": "See [S1]", "citations": ["S1"]}],
        "uncertainty_notes": []
    }
    res = validate_debug_response_citations(debug_dict, str(debug_dict), mock_citations)
    assert res.is_valid is False
    assert "S99" in res.invalid_source_ids

def test_debug_validator_unauthorized_citation_in_uncertainty_notes(mock_citations):
    debug_dict = {
        "error_summary": "Error summary",
        "likely_causes": [{"cause": "Known cause", "explanation": "See [S1]", "citations": ["S1"]}],
        "diagnostic_checks": [],
        "suggested_fixes": [],
        "uncertainty_notes": ["Note about docs [S99]"]
    }
    res = validate_debug_response_citations(debug_dict, str(debug_dict), mock_citations)
    assert res.is_valid is False
    assert "S99" in res.invalid_source_ids

def test_debug_validator_fabricated_url_in_diagnostic_checks(mock_citations):
    debug_dict = {
        "error_summary": "Error summary",
        "likely_causes": [{"cause": "Known cause", "explanation": "See [S1]", "citations": ["S1"]}],
        "diagnostic_checks": ["Check https://fastapi.tiangolo.com.evil.com/check/"],
        "suggested_fixes": [],
        "uncertainty_notes": []
    }
    res = validate_debug_response_citations(debug_dict, str(debug_dict), mock_citations)
    assert res.is_valid is False
    assert len(res.fabricated_urls) == 1

def test_debug_validator_malformed_citation_syntax(mock_citations):
    debug_dict = {
        "error_summary": "Error referencing [Source 1]",
        "likely_causes": [{"cause": "Cause", "explanation": "Explanation", "citations": ["S1"]}],
        "diagnostic_checks": [],
        "suggested_fixes": [],
        "uncertainty_notes": []
    }
    res = validate_debug_response_citations(debug_dict, str(debug_dict), mock_citations)
    assert res.is_valid is False
    assert any("Source 1" in sid for sid in res.invalid_source_ids)

def test_raw_debug_payload_schema_enforcement():
    # Valid payload parses cleanly
    valid_data = {
        "error_summary": "Valid summary",
        "likely_causes": [{"cause": "Cause", "explanation": "Explanation", "citations": ["S1"]}],
        "diagnostic_checks": ["Check 1"],
        "suggested_fixes": [{"fix": "Fix", "explanation": "Explanation", "citations": ["S1"]}],
        "uncertainty_notes": ["Note 1"]
    }
    payload = RawDebugPayload.model_validate(valid_data)
    assert payload.error_summary == "Valid summary"

    # Invalid: likely_causes contains raw strings instead of dicts
    invalid_causes = {
        "error_summary": "Valid summary",
        "likely_causes": ["Cause 1 as string"],
        "diagnostic_checks": ["Check 1"]
    }
    with pytest.raises(ValidationError):
        RawDebugPayload.model_validate(invalid_causes)

    # Invalid: diagnostic_checks contains ints instead of strings
    invalid_checks = {
        "error_summary": "Valid summary",
        "likely_causes": [],
        "diagnostic_checks": [12345]
    }
    with pytest.raises(ValidationError):
        RawDebugPayload.model_validate(invalid_checks)

    # Invalid: missing required error_summary
    missing_summary = {
        "likely_causes": [],
        "diagnostic_checks": []
    }
    with pytest.raises(ValidationError):
        RawDebugPayload.model_validate(missing_summary)
