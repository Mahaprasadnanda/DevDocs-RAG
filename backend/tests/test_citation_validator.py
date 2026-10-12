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
