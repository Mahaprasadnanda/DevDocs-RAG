import pytest
import math
import numpy as np
from unittest.mock import MagicMock
from app.retrieval.service import RetrievalService, IndexUnavailableError
from app.chunking.models import ChunkRecord
from app.retrieval.models import RetrievalResult

@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    def fake_encode(texts, batch_size):
        # Return 1 vector of dimension 4
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    embedder.encode.side_effect = fake_encode
    return embedder

@pytest.fixture
def mock_store():
    store = MagicMock()
    store.index = MagicMock()
    store.index.ntotal = 5
    
    def fake_search(query_embedding, k):
        distances = np.array([[0.9, 0.8, 0.7, 0.0, 0.0]])[:, :k]
        indices = np.array([[0, 2, 4, -1, -1]])[:, :k]
        return distances, indices
        
    store.index.search.side_effect = fake_search
    
    store.metadata = [
        ChunkRecord(
            chunk_id=f"chunk_{i}",
            document_id=f"doc_{i}",
            title=f"Title {i}",
            source_url=f"https://example.com/{i}",
            chunk_index=i,
            text=f"Text {i}"
        ) for i in range(5)
    ]
    
    return store

@pytest.fixture
def retrieval_service(mock_embedder, mock_store):
    service = RetrievalService()
    service.embedder = mock_embedder
    service.store = mock_store
    service._is_loaded = True # Mock loaded state
    return service

def test_retrieve_success(retrieval_service):
    results = retrieval_service.retrieve("query text", top_k=3)
    
    assert len(results) == 3
    assert results[0].chunk_id == "chunk_0"
    assert results[0].score == 0.9
    assert results[1].chunk_id == "chunk_2"
    assert results[1].score == 0.8
    assert results[2].chunk_id == "chunk_4"
    assert results[2].score == 0.7

def test_retrieve_empty_query(retrieval_service):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieval_service.retrieve("")
        
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieval_service.retrieve("   \n ")

def test_retrieve_invalid_top_k(retrieval_service):
    with pytest.raises(ValueError, match="positive integer"):
        retrieval_service.retrieve("query", top_k=0)
        
    with pytest.raises(ValueError, match="positive integer"):
        retrieval_service.retrieve("query", top_k=-5)

def test_retrieve_empty_index(retrieval_service):
    retrieval_service.store.index.ntotal = 0
    results = retrieval_service.retrieve("query", top_k=5)
    assert len(results) == 0

def test_retrieve_unavailable_index():
    service = RetrievalService()
    service.store = MagicMock()
    service.store.load.side_effect = FileNotFoundError("Missing")
    
    with pytest.raises(IndexUnavailableError):
        service.retrieve("query")

def test_retrieve_capping_at_ntotal(retrieval_service, mock_store):
    def fake_search_all(query_emb, k):
        assert k == 5 # Capped
        distances = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]])
        indices = np.array([[0, 1, 2, 3, 4]])
        return distances, indices
        
    mock_store.index.search.side_effect = fake_search_all
    
    results = retrieval_service.retrieve("query", top_k=10)
    assert len(results) == 5

def test_retrieve_faiss_sentinel(retrieval_service, mock_store):
    def fake_search_sentinel(query_emb, k):
        distances = np.array([[0.9, 0.0]])
        indices = np.array([[0, -1]])
        return distances, indices
    
    mock_store.index.search.side_effect = fake_search_sentinel
    results = retrieval_service.retrieve("query", top_k=2)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_0"

def test_retrieve_invalid_metadata_index(retrieval_service, mock_store):
    def fake_search_invalid(query_emb, k):
        distances = np.array([[0.9, 0.8]])
        indices = np.array([[0, 999]]) # 999 out of bounds
        return distances, indices
    
    mock_store.index.search.side_effect = fake_search_invalid
    results = retrieval_service.retrieve("query", top_k=2)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_0"

def test_retrieve_nonfinite_score(retrieval_service, mock_store):
    def fake_search_nonfinite(query_emb, k):
        distances = np.array([[np.nan, 0.8]])
        indices = np.array([[0, 1]])
        return distances, indices
    
    mock_store.index.search.side_effect = fake_search_nonfinite
    results = retrieval_service.retrieve("query", top_k=2)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_1"
