import os
import json
import pytest
from app.chunking.pipeline import run_chunking_pipeline
from app.core.config import settings

@pytest.fixture
def mock_directories(tmp_path):
    input_dir = tmp_path / "ingested"
    input_dir.mkdir()
    
    output_dir = tmp_path / "chunks"
    
    # Create a valid input document
    valid_doc = {
        "document_id": "test_id",
        "title": "Test Title",
        "source_url": "https://example.com",
        "canonical_url": None,
        "headings": [],
        "text": "This is a valid test document. " * 50,
        "fetched_at": "2026-10-10T12:00:00Z"
    }
    
    with open(input_dir / "valid.json", "w", encoding="utf-8") as f:
        json.dump(valid_doc, f)
        
    # Create a malformed input document
    with open(input_dir / "malformed.json", "w", encoding="utf-8") as f:
        f.write("{ invalid json")
        
    return input_dir, output_dir

def test_pipeline_processing(mock_directories, monkeypatch):
    input_dir, output_dir = mock_directories
    monkeypatch.setattr(settings, "INGESTION_OUTPUT_DIR", str(input_dir))
    monkeypatch.setattr(settings, "CHUNKING_OUTPUT_DIR", str(output_dir))
    
    results = run_chunking_pipeline()
    
    assert results["files_found"] == 2
    assert results["valid_documents_processed"] == 1
    assert results["documents_skipped_or_failed"] == 1
    assert results["total_chunks_generated"] > 0
    
    # Check output
    assert os.path.exists(output_dir)
    out_files = os.listdir(output_dir)
    assert len(out_files) == 1
    assert out_files[0] == "test_id_chunks.json"
    
    with open(os.path.join(output_dir, out_files[0]), "r", encoding="utf-8") as f:
        chunks = json.load(f)
        assert len(chunks) > 0
        assert chunks[0]["document_id"] == "test_id"
        assert chunks[0]["title"] == "Test Title"

def test_pipeline_missing_input_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "INGESTION_OUTPUT_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(settings, "CHUNKING_OUTPUT_DIR", str(tmp_path / "chunks"))
    
    results = run_chunking_pipeline()
    assert "error" in results

def test_pipeline_empty_input_dir(tmp_path, monkeypatch):
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    monkeypatch.setattr(settings, "INGESTION_OUTPUT_DIR", str(input_dir))
    monkeypatch.setattr(settings, "CHUNKING_OUTPUT_DIR", str(tmp_path / "chunks"))
    
    results = run_chunking_pipeline()
    assert results["files_found"] == 0
