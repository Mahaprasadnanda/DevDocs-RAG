import logging
import math
from typing import List, Optional
from app.retrieval.models import RetrievalResult
from app.vector_store.embeddings import EmbeddingService
from app.vector_store.index import VectorStore
from app.core.config import settings

logger = logging.getLogger(__name__)

class IndexUnavailableError(Exception):
    pass

class RetrievalService:
    def __init__(self):
        self.embedder = EmbeddingService(settings.EMBEDDING_MODEL_NAME)
        self.store = VectorStore(settings.INDEX_OUTPUT_DIR)
        self._is_loaded = False
        
    def _ensure_loaded(self):
        if not self._is_loaded:
            try:
                self.store.load()
                self._is_loaded = True
            except FileNotFoundError as e:
                logger.error(f"Failed to load index: {e}")
                raise IndexUnavailableError("Vector index is missing or incomplete.")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                raise IndexUnavailableError(f"Vector index could not be loaded: {e}")
                
    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        min_score: Optional[float] = None
    ) -> List[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
            
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
            
        self._ensure_loaded()
        
        if self.store.index is None or self.store.index.ntotal == 0:
            return []
            
        threshold = min_score if min_score is not None else settings.RETRIEVAL_MIN_SCORE
            
        # Ensure we don't request more than available
        k = min(top_k, self.store.index.ntotal)
        
        # Generate query embedding
        query_text = query.strip()
        query_embedding = self.embedder.encode([query_text], batch_size=1)
        
        # Search
        distances, indices = self.store.index.search(query_embedding, k)
        
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            score = float(distances[0][i])
            
            # FAISS returns -1 if it can't find enough results
            if idx == -1:
                continue
                
            if idx < 0 or idx >= len(self.store.metadata):
                logger.warning(f"Invalid FAISS index returned: {idx}")
                continue
                
            if not math.isfinite(score):
                logger.warning(f"Invalid FAISS score returned: {score}")
                continue
                
            if threshold is not None and score < threshold:
                logger.debug(f"Filtered chunk {idx} with score {score:.3f} below threshold {threshold}")
                continue
                
            chunk = self.store.metadata[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_url=chunk.source_url,
                    document_id=chunk.document_id,
                    section_title=chunk.section_heading,
                    score=score
                )
            )
            
        return results

retrieval_service = RetrievalService()
