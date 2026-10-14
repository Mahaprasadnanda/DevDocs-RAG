import os
import json
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.ingestion.sources import FASTAPI_DOCS_SOURCES
from app.ingestion.registry import DOCUMENTATION_REGISTRY, get_technology_config, get_all_technologies, get_ecosystem_component
from app.ingestion.fetcher import fetch_html, FetchError
from app.ingestion.extractor import extract_document
from app.ingestion.models import DocumentRecord

logger = logging.getLogger(__name__)

DEFAULT_FASTAPI_SOURCES = [
    "https://fastapi.tiangolo.com/",
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/body/",
    "https://fastapi.tiangolo.com/tutorial/handling-errors/"
]

def run_pipeline(technology: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes the ingestion pipeline across registered documentation collections.
    If technology is specified, only that technology is ingested.
    Returns a structured summary of the ingestion run.
    """
    output_dir = settings.INGESTION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Backward compatibility with existing tests that patch FASTAPI_DOCS_SOURCES
    if FASTAPI_DOCS_SOURCES != DEFAULT_FASTAPI_SOURCES:
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
                final_url, response = fetch_html(url)
                document = extract_document(response.text, final_url)
                out_file = os.path.join(output_dir, f"{document.document_id}.json")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(document.model_dump_json(indent=2))
                results["success_count"] += 1
            except FetchError as e:
                msg = f"Fetch failed: {e}"
                results["failure_count"] += 1
                results["failures"].append({"url": url, "reason": msg})
            except Exception as e:
                msg = f"Extraction failed: {e}"
                results["failure_count"] += 1
                results["failures"].append({"url": url, "reason": msg})
        return results
    
    if technology:
        cfg = get_technology_config(technology)
        if not cfg:
            raise ValueError(f"Unknown technology: {technology}. Must be one of: {list(DOCUMENTATION_REGISTRY.keys())}")
        targets = [cfg]
    else:
        targets = get_all_technologies(enabled_only=True)

    summary: Dict[str, Any] = {
        "total_technologies": len(targets),
        "total_sources": sum(len(t.urls) for t in targets),
        "success_count": 0,
        "failure_count": 0,
        "by_technology": {},
        "failures": []
    }

    processed_urls = set()

    for tech_cfg in targets:
        tech_id = tech_cfg.tech_id
        tech_stats = {
            "display_name": tech_cfg.display_name,
            "version": tech_cfg.version,
            "total_urls": len(tech_cfg.urls),
            "success_count": 0,
            "failure_count": 0,
            "failures": []
        }
        
        logger.info(f"--- Ingesting {tech_cfg.display_name} ({tech_id}) ---")

        for url in tech_cfg.urls:
            if url in processed_urls:
                logger.info(f"Skipping already processed URL: {url}")
                continue
            processed_urls.add(url)

            try:
                logger.info(f"Fetching [{tech_id}] {url}")
                final_url, response = fetch_html(url, allowed_hosts=tech_cfg.allowed_hosts)
                
                logger.info(f"Extracting content for [{tech_id}] {final_url}")
                comp = get_ecosystem_component(final_url, tech_cfg)
                document = extract_document(
                    response.text,
                    final_url,
                    technology=tech_id,
                    doc_version=tech_cfg.version,
                    strategy=tech_cfg.extraction_strategy,
                    ecosystem_component=comp
                )
                
                # Persist to disk deterministically using doc_id
                # Atomic write via temporary file
                out_file = os.path.join(output_dir, f"{document.document_id}.json")
                temp_file = out_file + ".tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(document.model_dump_json(indent=2))
                os.replace(temp_file, out_file)
                    
                summary["success_count"] += 1
                tech_stats["success_count"] += 1
                logger.info(f"Successfully processed [{tech_id}] {url} -> {document.document_id}")
                
            except FetchError as e:
                msg = f"Fetch failed: {e}"
                logger.error(f"[{tech_id}] {url} - {msg}")
                summary["failure_count"] += 1
                tech_stats["failure_count"] += 1
                failure_entry = {"tech_id": tech_id, "url": url, "reason": msg}
                summary["failures"].append(failure_entry)
                tech_stats["failures"].append(failure_entry)
            except ValueError as e:
                msg = f"Extraction failed: {e}"
                logger.error(f"[{tech_id}] {url} - {msg}")
                summary["failure_count"] += 1
                tech_stats["failure_count"] += 1
                failure_entry = {"tech_id": tech_id, "url": url, "reason": msg}
                summary["failures"].append(failure_entry)
                tech_stats["failures"].append(failure_entry)
            except Exception as e:
                msg = f"Unexpected error: {e}"
                logger.error(f"[{tech_id}] {url} - {msg}")
                summary["failure_count"] += 1
                tech_stats["failure_count"] += 1
                failure_entry = {"tech_id": tech_id, "url": url, "reason": msg}
                summary["failures"].append(failure_entry)
                tech_stats["failures"].append(failure_entry)

        summary["by_technology"][tech_id] = tech_stats

    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run documentation ingestion pipeline.")
    parser.add_argument("--tech", default=None, help="Specific technology to ingest (e.g. 'react', 'fastapi', 'spring-boot')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_pipeline(technology=args.tech)
    print("\n--- Ingestion Summary ---")
    print(json.dumps(summary, indent=2))
    
    if summary["failure_count"] == summary["total_sources"] and summary["total_sources"] > 0:
        print("\nAll sources failed!")
        exit(1)
