import os
import pytest
import numpy as np
from app.vector_store.index import VectorStore
from app.chunking.models import ChunkRecord

@pytest.fixture
def dummy_chunks():
    return [
        ChunkRecord(
            chunk_id=f"chunk_{i}",
            document_id="doc_1",
            title="Title",
            source_url="https://example.com",
            chunk_index=i,
            text=f"Text {i}"
        ) for i in range(5)
    ]

@pytest.fixture
def dummy_embeddings():
    vecs = np.random.rand(5, 128).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def test_vector_store_build_and_save(tmp_path, dummy_chunks, dummy_embeddings):
    index_dir = str(tmp_path / "index")
    store = VectorStore(index_dir)
    
    store.build(dummy_embeddings, dummy_chunks, "test-model")
    
    assert store.dimension == 128
    assert store.index.ntotal == 5
    
    store.save()
    
    assert os.path.exists(store.index_path)
    assert os.path.exists(store.metadata_path)
    assert os.path.exists(store.manifest_path)

def test_vector_store_load(tmp_path, dummy_chunks, dummy_embeddings):
    index_dir = str(tmp_path / "index")
    store1 = VectorStore(index_dir)
    store1.build(dummy_embeddings, dummy_chunks, "test-model")
    store1.save()
    
    # Reload
    store2 = VectorStore(index_dir)
    store2.load()
    
    assert store2.dimension == 128
    assert store2.model_name == "test-model"
    assert store2.index.ntotal == 5
    assert len(store2.metadata) == 5
    assert store2.metadata[0].chunk_id == "chunk_0"

def test_vector_store_mismatch(tmp_path, dummy_chunks, dummy_embeddings):
    index_dir = str(tmp_path / "index")
    store = VectorStore(index_dir)
    
    # Mismatch counts
    with pytest.raises(ValueError, match="Mismatched counts"):
        store.build(dummy_embeddings[:3], dummy_chunks, "test-model")

def test_vector_store_missing_files(tmp_path):
    index_dir = str(tmp_path / "index")
    store = VectorStore(index_dir)
    with pytest.raises(FileNotFoundError):
        store.load()
        
def test_vector_store_cosine_similarity(tmp_path, dummy_chunks):
    index_dir = str(tmp_path / "index")
    store = VectorStore(index_dir)
    
    # Create normalized vectors where vector 0 matches a query exactly
    vecs = np.zeros((5, 128), dtype=np.float32)
    vecs[0, 0] = 1.0  # Unit length along first axis
    vecs[1, 1] = 1.0  # Unit length along second axis
    
    store.build(vecs, dummy_chunks, "test-model")
    
    # Query vector matching vector 0
    query = np.zeros((1, 128), dtype=np.float32)
    query[0, 0] = 1.0
    
    # Use exact inner product (which is cosine similarity for normalized vectors)
    distances, indices = store.index.search(query, k=1)
    
    # Should find vector 0 with similarity ~1.0
    assert indices[0][0] == 0
    assert np.isclose(distances[0][0], 1.0)
