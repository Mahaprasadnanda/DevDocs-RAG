import os
import json
import logging
from typing import Dict, Any, List
from app.core.config import settings
from app.chunking.models import ChunkRecord
from app.vector_store.embeddings import EmbeddingService
from app.vector_store.index import VectorStore

logger = logging.getLogger(__name__)

def run_indexing_pipeline() -> Dict[str, Any]:
    chunk_dir = settings.CHUNKING_OUTPUT_DIR
    index_dir = settings.INDEX_OUTPUT_DIR
    
    if not os.path.exists(chunk_dir):
        logger.error(f"Chunk directory not found: {chunk_dir}. Please run the chunking pipeline first.")
        return {"error": "Chunk directory not found."}
        
    chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith(".json")]
    if not chunk_files:
        logger.error("No chunk files found.")
        return {"error": "No chunk files found."}
        
    all_chunks: List[ChunkRecord] = []
    
    for filename in chunk_files:
        path = os.path.join(chunk_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            for item in data:
                all_chunks.append(ChunkRecord(**item))
        except Exception as e:
            logger.error(f"Failed to load chunks from {filename}: {e}")
            
    if not all_chunks:
        return {"error": "No valid chunk records parsed."}
        
    # Extract texts for embedding, preserving exact order
    texts = [c.text for c in all_chunks]
    
    # Initialize embedding service
    embedder = EmbeddingService(model_name=settings.EMBEDDING_MODEL_NAME)
    
    try:
        embeddings = embedder.encode(texts, batch_size=settings.EMBEDDING_BATCH_SIZE)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return {"error": f"Failed to generate embeddings: {e}"}
        
    # Build and save index
    store = VectorStore(index_dir)
    try:
        store.build(embeddings, all_chunks, model_name=settings.EMBEDDING_MODEL_NAME)
        store.save()
    except Exception as e:
        logger.error(f"Failed to build or save index: {e}")
        return {"error": f"Failed to build or save index: {e}"}
        
    tech_counts = {}
    for c in all_chunks:
        t = getattr(c, "technology", "fastapi")
        tech_counts[t] = tech_counts.get(t, 0) + 1

    return {
        "input_chunk_count": len(all_chunks),
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": store.dimension,
        "vectors_indexed": store.index.ntotal,
        "technologies_indexed": sorted(list(tech_counts.keys())),
        "by_technology": tech_counts,
        "index_output_path": store.index_path,
        "metadata_output_path": store.metadata_path,
        "status": "success"
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_indexing_pipeline()
    print("\n--- Indexing Summary ---")
    print(json.dumps(summary, indent=2))
    
    if "error" in summary:
        print("\nIndexing failed.")
        exit(1)
