import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.vector_store.embeddings import EmbeddingService

@pytest.fixture
def mock_sentence_transformer():
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_instance = MagicMock()
        mock_st.return_value = mock_instance
        # Fake dimension 384
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        
        # Fake encode returning normalized float32
        def fake_encode(texts, batch_size, convert_to_numpy, normalize_embeddings):
            n = len(texts)
            # Create some dummy vectors
            vecs = np.random.rand(n, 384).astype(np.float32)
            # Normalize them
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms
            
        mock_instance.encode.side_effect = fake_encode
        yield mock_instance

def test_embedding_service_basic(mock_sentence_transformer):
    service = EmbeddingService("dummy-model")
    assert service.dimension == 384
    
    texts = ["hello world", "test sentence"]
    embeddings = service.encode(texts)
    
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32
    
    # Check unit normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0)
    
    # Verify mock was called correctly
    mock_sentence_transformer.encode.assert_called_once_with(
        texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
    )

def test_embedding_service_empty(mock_sentence_transformer):
    service = EmbeddingService("dummy-model")
    embeddings = service.encode([])
    assert embeddings.shape == (0, 384)

def test_embedding_service_zero_norm():
    # Test strict explicit normalization fallback for safety
    service = EmbeddingService("dummy-model")
    service._dimension = 4
    # Mocking the _model directly
    service._model = MagicMock()
    # Provide a batch with a zero vector
    service._model.encode.return_value = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ])
    
    embeddings = service.encode(["empty", "real"])
    
    # First should remain 0, second should be normalized
    assert np.allclose(embeddings[0], 0.0)
    assert np.allclose(embeddings[1], [1.0, 0.0, 0.0, 0.0])

def test_embedding_service_invalid_values():
    service = EmbeddingService("dummy-model")
    service._dimension = 4
    service._model = MagicMock()
    service._model.encode.return_value = np.array([
        [np.nan, 0.0, 0.0, 0.0]
    ])
    
    with pytest.raises(ValueError, match="NaN"):
        service.encode(["nan text"])
