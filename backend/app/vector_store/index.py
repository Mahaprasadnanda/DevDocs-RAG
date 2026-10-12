import os
import json
import logging
from typing import List, Optional
from app.chunking.models import ChunkRecord
import numpy as np

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "vector.index")
        self.metadata_path = os.path.join(index_dir, "metadata.json")
        self.manifest_path = os.path.join(index_dir, "manifest.json")
        
        self.index = None  # faiss.Index
        self.metadata: List[ChunkRecord] = []
        self.dimension: Optional[int] = None
        self.model_name: Optional[str] = None
        
    def build(self, embeddings: np.ndarray, chunks: List[ChunkRecord], model_name: str):
        """Builds a new index from scratch."""
        import faiss
        
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatched counts: {len(embeddings)} vectors, {len(chunks)} chunks.")
            
        if len(embeddings) == 0:
            raise ValueError("Cannot build index with 0 vectors.")
            
        self.dimension = embeddings.shape[1]
        self.model_name = model_name
        
        # Use IndexFlatIP for exact inner-product (cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.metadata = chunks
        
    def save(self):
        """Persists the index, metadata, and manifest securely."""
        import faiss
        
        if self.index is None:
            raise RuntimeError("No index to save.")
            
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Write metadata
        temp_meta = self.metadata_path + ".tmp"
        with open(temp_meta, "w", encoding="utf-8") as f:
            json.dump([json.loads(c.model_dump_json()) for c in self.metadata], f, indent=2)
            
        # Write index
        temp_idx = self.index_path + ".tmp"
        faiss.write_index(self.index, temp_idx)
        
        # Compute per-technology statistics
        tech_counts = {}
        for c in self.metadata:
            t = getattr(c, "technology", "fastapi")
            tech_counts[t] = tech_counts.get(t, 0) + 1

        # Write manifest for consistency checks
        manifest = {
            "version": "1.0",
            "model_name": self.model_name,
            "dimension": self.dimension,
            "vector_count": self.index.ntotal,
            "metadata_count": len(self.metadata),
            "technologies": sorted(list(tech_counts.keys())),
            "technology_counts": tech_counts
        }
        
        temp_manifest = self.manifest_path + ".tmp"
        with open(temp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            
        # Atomic replacement
        os.replace(temp_meta, self.metadata_path)
        os.replace(temp_idx, self.index_path)
        os.replace(temp_manifest, self.manifest_path)
        logger.info(f"Successfully saved index with {self.index.ntotal} vectors to {self.index_dir}")
        
    def load(self):
        """Loads index and metadata, validating consistency."""
        import faiss
        
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError("Manifest file not found. Index might not exist.")
            
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            
        self.model_name = manifest["model_name"]
        self.dimension = manifest["dimension"]
        
        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError("Metadata file not found.")
            
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            self.metadata = [ChunkRecord(**m) for m in meta_data]
            
        if len(self.metadata) != manifest["metadata_count"]:
            raise ValueError("Metadata count mismatch with manifest.")
            
        # Load FAISS index
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("Vector index file not found.")
            
        self.index = faiss.read_index(self.index_path)
        
        if self.index.ntotal != manifest["vector_count"]:
            raise ValueError("Index vector count mismatch with manifest.")
            
        if self.index.d != self.dimension:
            raise ValueError("Index dimension mismatch with manifest.")
            
        if self.index.ntotal != len(self.metadata):
            raise ValueError("Vector count does not match metadata count.")
            
        logger.info(f"Successfully loaded index with {self.index.ntotal} vectors.")
