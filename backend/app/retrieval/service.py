import logging
import math
from typing import List, Optional
from app.retrieval.models import RetrievalResult
from app.vector_store.embeddings import EmbeddingService
from app.vector_store.index import VectorStore
from app.core.config import settings

from app.retrieval.tech_detector import detect_technology
from app.ingestion.registry import get_technology_config

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
        min_score: Optional[float] = None,
        technology: Optional[str] = None,
        version: Optional[str] = None
    ) -> List[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
            
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
            
        self._ensure_loaded()
        
        if self.store.index is None or self.store.index.ntotal == 0:
            return []
            
        threshold = min_score if min_score is not None else settings.RETRIEVAL_MIN_SCORE

        # Resolve technology filter
        target_techs: Optional[List[str]] = None
        if technology and technology.strip().lower() not in ["auto", "all"]:
            cfg = get_technology_config(technology.strip())
            if cfg:
                target_techs = [cfg.tech_id]
            else:
                target_techs = [technology.strip().lower()]
        elif technology is not None and technology.strip().lower() == "auto":
            detected = detect_technology(query=query)
            if detected:
                target_techs = detected

        # When filtering by technology/version, retrieve a larger candidate pool from FAISS
        # to ensure filtered results are not crowded out
        if target_techs or version:
            k_search = min(max(top_k * 6, 60), self.store.index.ntotal)
        else:
            k_search = min(top_k, self.store.index.ntotal)
        
        # Generate query embedding
        query_text = query.strip()
        query_embedding = self.embedder.encode([query_text], batch_size=1)
        
        # Search FAISS
        distances, indices = self.store.index.search(query_embedding, k_search)
        
        results = []
        num_candidates = min(k_search, len(indices[0]))
        for i in range(num_candidates):
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
            chunk_tech = getattr(chunk, "technology", "fastapi")
            chunk_ver = getattr(chunk, "doc_version", None)

            # Apply technology filter if specified
            if target_techs:
                if chunk_tech not in target_techs:
                    continue

            # Apply version filter if specified
            if version and chunk_ver:
                if chunk_ver.lower() != version.lower():
                    continue

            chunk_comp = getattr(chunk, "ecosystem_component", None)

            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_url=chunk.source_url,
                    document_id=chunk.document_id,
                    section_title=chunk.section_heading,
                    score=score,
                    technology=chunk_tech,
                    doc_version=chunk_ver,
                    ecosystem_component=chunk_comp
                )
            )

            if len(results) >= top_k:
                break
                
        return results

    def get_indexed_technologies(self) -> List[str]:
        """Returns list of unique technologies currently present in the vector store."""
        self._ensure_loaded()
        techs = set()
        for c in self.store.metadata:
            techs.add(getattr(c, "technology", "fastapi"))
        return sorted(list(techs))

retrieval_service = RetrievalService()
