import os
import json
import pytest
from unittest.mock import patch, MagicMock
from app.ingestion.pipeline import run_pipeline
from app.core.config import settings

@pytest.fixture
def mock_fetch():
    with patch("app.ingestion.pipeline.fetch_html") as mock_fetch_html:
        yield mock_fetch_html

@pytest.fixture
def mock_extract():
    with patch("app.ingestion.pipeline.extract_document") as mock_extract_document:
        yield mock_extract_document

@pytest.fixture
def mock_sources():
    with patch("app.ingestion.pipeline.FASTAPI_DOCS_SOURCES", ["url1", "url2", "url1"]): # Duplicate to test deduplication
        yield

def test_pipeline_success(mock_fetch, mock_extract, mock_sources, tmp_path):
    settings.INGESTION_OUTPUT_DIR = str(tmp_path)
    
    mock_response = MagicMock()
    mock_response.text = "<html></html>"
    mock_fetch.return_value = ("resolved_url", mock_response)
    
    mock_doc = MagicMock()
    mock_doc.document_id = "test_id"
    mock_doc.model_dump_json.return_value = '{"id": "test_id"}'
    mock_extract.return_value = mock_doc
    
    results = run_pipeline()
    
    assert results["total_sources"] == 3
    # Wait, the deduplication happens, so it only processes 2 unique URLs!
    assert results["success_count"] == 2
    assert results["failure_count"] == 0
    assert len(results["failures"]) == 0
    
    assert mock_fetch.call_count == 2
    assert mock_extract.call_count == 2
    
    # Check that file was created
    assert os.path.exists(os.path.join(tmp_path, "test_id.json"))

def test_pipeline_failures(mock_fetch, mock_extract, mock_sources, tmp_path):
    settings.INGESTION_OUTPUT_DIR = str(tmp_path)
    
    from app.ingestion.fetcher import FetchError
    
    # Fail first URL, succeed second
    def fetch_side_effect(url):
        if url == "url1":
            raise FetchError("Failed")
        return ("resolved_url", MagicMock())
        
    mock_fetch.side_effect = fetch_side_effect
    
    mock_doc = MagicMock()
    mock_doc.document_id = "test_id_2"
    mock_doc.model_dump_json.return_value = "{}"
    mock_extract.return_value = mock_doc
    
    results = run_pipeline()
    
    assert results["success_count"] == 1
    assert results["failure_count"] == 1
    assert len(results["failures"]) == 1
    assert results["failures"][0]["url"] == "url1"

def test_pipeline_all_fail(mock_fetch, mock_sources, tmp_path):
    settings.INGESTION_OUTPUT_DIR = str(tmp_path)
    from app.ingestion.fetcher import FetchError
    mock_fetch.side_effect = FetchError("Failed")
    
    results = run_pipeline()
    assert results["success_count"] == 0
    assert results["failure_count"] == 2
