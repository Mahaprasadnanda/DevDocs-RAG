import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.vector_store.pipeline import run_indexing_pipeline
from app.core.config import settings

@pytest.fixture
def mock_directories(tmp_path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    
    index_dir = tmp_path / "index"
    
    # Create valid chunks
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "document_id": "doc_1",
            "title": "Title",
            "source_url": "https://example.com",
            "chunk_index": i,
            "text": f"Text {i}"
        } for i in range(2)
    ]
    
    with open(chunk_dir / "doc1_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)
        
    return chunk_dir, index_dir

@patch("app.vector_store.pipeline.EmbeddingService")
def test_indexing_pipeline(mock_emb_class, mock_directories, monkeypatch):
    chunk_dir, index_dir = mock_directories
    monkeypatch.setattr(settings, "CHUNKING_OUTPUT_DIR", str(chunk_dir))
    monkeypatch.setattr(settings, "INDEX_OUTPUT_DIR", str(index_dir))
    monkeypatch.setattr(settings, "EMBEDDING_MODEL_NAME", "dummy")
    
    mock_emb_instance = MagicMock()
    # Return normalized vectors
    vecs = np.random.rand(2, 64).astype(np.float32)
    mock_emb_instance.encode.return_value = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    mock_emb_class.return_value = mock_emb_instance
    
    summary = run_indexing_pipeline()
    
    assert summary.get("status") == "success"
    assert summary["input_chunk_count"] == 2
    assert summary["embedding_dimension"] == 64
    assert summary["vectors_indexed"] == 2
    
    # Check index existence
    assert os.path.exists(index_dir / "vector.index")
