import os
import json
import logging
from typing import Dict, Any
from app.core.config import settings
from app.ingestion.models import DocumentRecord
from app.chunking.chunker import process_document

logger = logging.getLogger(__name__)

def run_chunking_pipeline() -> Dict[str, Any]:
    input_dir = settings.INGESTION_OUTPUT_DIR
    output_dir = settings.CHUNKING_OUTPUT_DIR
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}. Please run the ingestion pipeline first.")
        return {"error": "Input directory not found."}
        
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "files_found": 0,
        "valid_documents_processed": 0,
        "documents_skipped_or_failed": 0,
        "total_chunks_generated": 0,
        "output_path": output_dir
    }
    
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    results["files_found"] = len(files)
    
    if not files:
        logger.info("No ingested documents found to process.")
        return results
        
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Validate schema
            doc = DocumentRecord(**data)
            
            # Generate chunks
            chunks = process_document(
                doc,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            
            # Persist chunks as JSON array
            # Decision: One JSON file per document containing its array of chunks.
            # Rationale: Matches the ingestion pattern of one file per document, makes it 
            # easy to inspect and guarantees deterministic file paths per source doc.
            out_filename = f"{doc.document_id}_chunks.json"
            out_file = os.path.join(output_dir, out_filename)
            
            # Atomic write via temp file
            temp_out_file = out_file + ".tmp"
            with open(temp_out_file, "w", encoding="utf-8") as f:
                json.dump([json.loads(c.model_dump_json()) for c in chunks], f, indent=2)
            os.replace(temp_out_file, out_file)
                
            tech = getattr(doc, "technology", "fastapi")
            if "by_technology" not in results:
                results["by_technology"] = {}
            if tech not in results["by_technology"]:
                results["by_technology"][tech] = {"documents": 0, "chunks": 0}
            results["by_technology"][tech]["documents"] += 1
            results["by_technology"][tech]["chunks"] += len(chunks)

            results["valid_documents_processed"] += 1
            results["total_chunks_generated"] += len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            results["documents_skipped_or_failed"] += 1
            
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_chunking_pipeline()
    print("\n--- Chunking Summary ---")
    print(json.dumps(summary, indent=2))
    
    if "error" in summary or (summary.get("files_found", 0) > 0 and summary.get("valid_documents_processed", 0) == 0):
        print("\nChunking failed or no valid documents were processed.")
        exit(1)
