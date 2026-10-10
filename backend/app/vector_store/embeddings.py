import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._dimension: Optional[int] = None
        
    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            # Lazy import to avoid loading heavy modules if not needed
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self._dimension}")
            
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension
        
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates normalized float32 embeddings for a list of texts.
        """
        if not texts:
            # Return empty array with correct dimension if dimension is known, else we have to load it
            return np.empty((0, self.dimension), dtype=np.float32)
            
        self._load_model()
        
        embeddings = self._model.encode(
            texts, 
            batch_size=batch_size, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
        # Ensure float32 (FAISS requirement)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        # Validate values
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or infinite values.")
            
        # Ensure unit length mathematically for all non-zero vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid divide by zero for zero-vectors
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms
        
        return embeddings
