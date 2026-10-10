import os
import json
import logging
from typing import List, Dict, Any
from app.core.config import settings
from app.ingestion.sources import FASTAPI_DOCS_SOURCES
from app.ingestion.fetcher import fetch_html, FetchError
from app.ingestion.extractor import extract_document
from app.ingestion.models import DocumentRecord

logger = logging.getLogger(__name__)

def run_pipeline() -> Dict[str, Any]:
    """
    Executes the ingestion pipeline on the curated sources.
    Returns a summary of the run.
    """
    output_dir = settings.INGESTION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "total_sources": len(FASTAPI_DOCS_SOURCES),
        "success_count": 0,
        "failure_count": 0,
        "failures": []
    }
    
    processed_urls = set()
    
    for url in FASTAPI_DOCS_SOURCES:
        if url in processed_urls:
            continue
        processed_urls.add(url)
        
        try:
            logger.info(f"Fetching {url}")
            final_url, response = fetch_html(url)
            
            logger.info(f"Extracting content for {url}")
            document = extract_document(response.text, final_url)
            
            # Persist to disk
            out_file = os.path.join(output_dir, f"{document.document_id}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(document.model_dump_json(indent=2))
                
            results["success_count"] += 1
            logger.info(f"Successfully processed {url}")
            
        except FetchError as e:
            msg = f"Fetch failed: {e}"
            logger.error(f"{url} - {msg}")
            results["failure_count"] += 1
            results["failures"].append({"url": url, "reason": msg})
        except ValueError as e:
            msg = f"Extraction failed: {e}"
            logger.error(f"{url} - {msg}")
            results["failure_count"] += 1
            results["failures"].append({"url": url, "reason": msg})
        except Exception as e:
            msg = f"Unexpected error: {e}"
            logger.error(f"{url} - {msg}")
            results["failure_count"] += 1
            results["failures"].append({"url": url, "reason": msg})
            
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_pipeline()
    print("\n--- Ingestion Summary ---")
    print(json.dumps(summary, indent=2))
    
    if summary["failure_count"] == summary["total_sources"] and summary["total_sources"] > 0:
        print("\nAll sources failed!")
        exit(1)
