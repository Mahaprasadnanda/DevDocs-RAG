import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure backend root is on sys.path
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.core.config import settings
from app.retrieval.service import retrieval_service


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate the given percentile (0 to 100) from a list of floats."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (percentile / 100.0)
    f = int(k)
    c = f + 1
    if c < len(sorted_vals):
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])
    return sorted_vals[f]


def run_evaluation(
    dataset_path: str = "data/eval/retrieval_eval_dataset.json",
    output_path: str = "data/eval/retrieval_eval_results.json",
    top_k: int = 8,
    min_score: Optional[float] = None
) -> Dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.is_absolute():
        dataset_file = backend_root / dataset_path

    if not dataset_file.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_file}")

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Use configured default if not explicitly specified
    effective_min_score = min_score if min_score is not None else settings.RETRIEVAL_MIN_SCORE

    print("================================================================================")
    print("           DevDocs RAG — Semantic Retrieval Evaluation Benchmark                ")
    print("================================================================================")
    print(f"Dataset path:           {dataset_file}")
    print(f"Total test queries:     {len(dataset)}")
    print(f"Retrieval top_k:        {top_k}")
    print(f"Relevance cutoff score: {effective_min_score}")
    print(f"Embedding model:        {settings.EMBEDDING_MODEL_NAME}")
    print(f"Index directory:        {settings.INDEX_OUTPUT_DIR}")
    print("--------------------------------------------------------------------------------\n")

    query_results: List[Dict[str, Any]] = []
    e2e_latencies: List[float] = []
    embed_latencies: List[float] = []
    search_latencies: List[float] = []

    in_domain_count = 0
    out_of_domain_count = 0
    recall_at_1_count = 0
    recall_at_3_count = 0
    recall_at_5_count = 0
    reciprocal_ranks: List[float] = []
    abstention_success_count = 0

    # Ensure index is loaded
    retrieval_service._ensure_loaded()
    manifest_path = Path(retrieval_service.store.manifest_path)
    index_manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as mf:
            index_manifest = json.load(mf)

    for item in dataset:
        qid = item["id"]
        query = item["query"]
        category = item.get("category", "general")
        is_answerable = item.get("answerable", True)
        expected_doc_ids = set(item.get("expected_document_ids", []))
        expected_urls = set(item.get("expected_urls", []))

        # 1. Measure query embedding latency
        t0 = time.perf_counter()
        query_vector = retrieval_service.embedder.encode([query.strip()], batch_size=1)
        t_embed = (time.perf_counter() - t0) * 1000.0  # ms

        # 2. Measure FAISS search latency
        k_search = min(top_k, retrieval_service.store.index.ntotal)
        t1 = time.perf_counter()
        distances, indices = retrieval_service.store.index.search(query_vector, k_search)
        t_search = (time.perf_counter() - t1) * 1000.0  # ms

        # 3. Overall retrieval with filtering
        t_start = time.perf_counter()
        retrieved_chunks = retrieval_service.retrieve(query, top_k=top_k, min_score=effective_min_score)
        t_total = (time.perf_counter() - t_start) * 1000.0  # ms

        e2e_latencies.append(t_total)
        embed_latencies.append(t_embed)
        search_latencies.append(t_search)

        # Document-level ranking: collapse multiple chunks from the same document
        unique_retrieved_docs: List[Dict[str, Any]] = []
        seen_doc_ids = set()

        for chunk in retrieved_chunks:
            if chunk.document_id not in seen_doc_ids:
                seen_doc_ids.add(chunk.document_id)
                unique_retrieved_docs.append({
                    "document_id": chunk.document_id,
                    "source_url": chunk.source_url,
                    "score": chunk.score
                })

        # Evaluate match
        first_relevant_rank: Optional[int] = None
        for rank_idx, doc_info in enumerate(unique_retrieved_docs, start=1):
            doc_id_match = doc_info["document_id"] in expected_doc_ids
            url_match = str(doc_info["source_url"]).rstrip("/") in {u.rstrip("/") for u in expected_urls}
            if doc_id_match or url_match:
                if first_relevant_rank is None:
                    first_relevant_rank = rank_idx

        q_res = {
            "id": qid,
            "query": query,
            "category": category,
            "answerable": is_answerable,
            "latency_ms": round(t_total, 2),
            "embed_ms": round(t_embed, 2),
            "search_ms": round(t_search, 2),
            "chunks_retrieved": len(retrieved_chunks),
            "unique_docs_retrieved": len(unique_retrieved_docs),
            "top_doc_score": retrieved_chunks[0].score if retrieved_chunks else None,
            "first_relevant_rank": first_relevant_rank,
            "retrieved_docs": [d["document_id"] for d in unique_retrieved_docs]
        }

        if is_answerable:
            in_domain_count += 1
            if first_relevant_rank == 1:
                recall_at_1_count += 1
            if first_relevant_rank is not None and first_relevant_rank <= 3:
                recall_at_3_count += 1
            if first_relevant_rank is not None and first_relevant_rank <= 5:
                recall_at_5_count += 1

            rr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0
            reciprocal_ranks.append(rr)
            q_res["reciprocal_rank"] = round(rr, 4)
        else:
            out_of_domain_count += 1
            # Out-of-domain query successfully abstained if 0 chunks passed threshold
            is_abstained = len(retrieved_chunks) == 0
            if is_abstained:
                abstention_success_count += 1
            q_res["abstained_correctly"] = is_abstained

        query_results.append(q_res)

    # Compute aggregate metrics
    recall_at_1 = (recall_at_1_count / in_domain_count) if in_domain_count > 0 else 0.0
    recall_at_3 = (recall_at_3_count / in_domain_count) if in_domain_count > 0 else 0.0
    recall_at_5 = (recall_at_5_count / in_domain_count) if in_domain_count > 0 else 0.0
    mrr = (sum(reciprocal_ranks) / in_domain_count) if in_domain_count > 0 else 0.0
    abstention_rate = (abstention_success_count / out_of_domain_count) if out_of_domain_count > 0 else 0.0

    latency_p50 = calculate_percentile(e2e_latencies, 50)
    latency_p95 = calculate_percentile(e2e_latencies, 95)
    latency_mean = sum(e2e_latencies) / len(e2e_latencies) if e2e_latencies else 0.0

    embed_p50 = calculate_percentile(embed_latencies, 50)
    embed_p95 = calculate_percentile(embed_latencies, 95)

    search_p50 = calculate_percentile(search_latencies, 50)
    search_p95 = calculate_percentile(search_latencies, 95)

    # Measure index file sizes on disk
    index_files = {}
    total_index_bytes = 0
    index_dir_path = Path(settings.INDEX_OUTPUT_DIR)
    if not index_dir_path.is_absolute():
        index_dir_path = backend_root / settings.INDEX_OUTPUT_DIR
    if index_dir_path.exists():
        for f in index_dir_path.iterdir():
            if f.is_file():
                size = f.stat().st_size
                index_files[f.name] = size
                total_index_bytes += size

    # Compute per-technology metrics
    tech_stats: Dict[str, Dict[str, Any]] = {}
    for item, q_res in zip(dataset, query_results):
        t = item.get("technology", "fastapi")
        if t not in tech_stats:
            tech_stats[t] = {
                "queries": 0,
                "in_domain": 0,
                "recall_1": 0,
                "recall_3": 0,
                "recall_5": 0,
                "rr_sum": 0.0,
                "out_of_domain": 0,
                "abstained": 0
            }
        stats = tech_stats[t]
        stats["queries"] += 1
        if item.get("answerable", True):
            stats["in_domain"] += 1
            if q_res.get("first_relevant_rank") == 1:
                stats["recall_1"] += 1
            if q_res.get("first_relevant_rank") is not None and q_res["first_relevant_rank"] <= 3:
                stats["recall_3"] += 1
            if q_res.get("first_relevant_rank") is not None and q_res["first_relevant_rank"] <= 5:
                stats["recall_5"] += 1
            stats["rr_sum"] += q_res.get("reciprocal_rank", 0.0)
        else:
            stats["out_of_domain"] += 1
            if q_res.get("abstained_correctly"):
                stats["abstained"] += 1

    by_tech_summary = {}
    for t, s in tech_stats.items():
        in_d = s["in_domain"]
        by_tech_summary[t] = {
            "queries": s["queries"],
            "recall_at_1": round(s["recall_1"] / in_d, 4) if in_d > 0 else None,
            "recall_at_3": round(s["recall_3"] / in_d, 4) if in_d > 0 else None,
            "recall_at_5": round(s["recall_5"] / in_d, 4) if in_d > 0 else None,
            "mrr": round(s["rr_sum"] / in_d, 4) if in_d > 0 else None,
            "abstention_rate": round(s["abstained"] / s["out_of_domain"], 4) if s["out_of_domain"] > 0 else None
        }

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_size": len(dataset),
        "in_domain_queries": in_domain_count,
        "out_of_domain_queries": out_of_domain_count,
        "retrieval_parameters": {
            "top_k": top_k,
            "min_score": effective_min_score,
            "metric_level": "document-level (collapsed chunks)"
        },
        "retrieval_quality": {
            "recall_at_1": round(recall_at_1, 4),
            "recall_at_3": round(recall_at_3, 4),
            "recall_at_5": round(recall_at_5, 4),
            "mean_reciprocal_rank_mrr": round(mrr, 4),
            "in_domain_top1_matches": recall_at_1_count,
            "in_domain_top3_matches": recall_at_3_count,
            "in_domain_top5_matches": recall_at_5_count
        },
        "by_technology": by_tech_summary,
        "abstention_quality": {
            "out_of_domain_total": out_of_domain_count,
            "out_of_domain_abstained": abstention_success_count,
            "abstention_rate": round(abstention_rate, 4)
        },
        "latency_ms": {
            "end_to_end": {
                "mean": round(latency_mean, 2),
                "median_p50": round(latency_p50, 2),
                "p95": round(latency_p95, 2)
            },
            "embedding_only": {
                "median_p50": round(embed_p50, 2),
                "p95": round(embed_p95, 2)
            },
            "faiss_search_only": {
                "median_p50": round(search_p50, 2),
                "p95": round(search_p95, 2)
            }
        },
        "index_statistics": {
            "model_name": settings.EMBEDDING_MODEL_NAME,
            "vector_dimension": index_manifest.get("dimension", 384) if index_manifest else 384,
            "total_vectors": index_manifest.get("vector_count", len(retrieval_service.store.metadata)) if index_manifest else len(retrieval_service.store.metadata),
            "total_documents": index_manifest.get("total_documents", len(retrieval_service.store.metadata)),
            "technologies": index_manifest.get("technologies", []),
            "index_disk_bytes": total_index_bytes,
            "index_files": index_files
        }
    }

    results_data = {
        "summary": summary,
        "query_results": query_results
    }

    # Write machine-readable JSON
    output_file = Path(output_path)
    if not output_file.is_absolute():
        output_file = backend_root / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    # Print human-readable summary
    print("--------------------------------------------------------------------------------")
    print("                           EVALUATION SUMMARY RESULTS                           ")
    print("--------------------------------------------------------------------------------")
    print(f"Evaluated Queries:          {len(dataset)} total ({in_domain_count} in-domain, {out_of_domain_count} out-of-domain)")
    print(f"Recall@1:                   {recall_at_1 * 100:.2f}% ({recall_at_1_count}/{in_domain_count})")
    print(f"Recall@3:                   {recall_at_3 * 100:.2f}% ({recall_at_3_count}/{in_domain_count})")
    print(f"Recall@5:                   {recall_at_5 * 100:.2f}% ({recall_at_5_count}/{in_domain_count})")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Abstention Filter Rate:     {abstention_rate * 100:.2f}% ({abstention_success_count}/{out_of_domain_count})")
    print("--------------------------------------------------------------------------------")
    print("PER-TECHNOLOGY BREAKDOWN:")
    for tech, t_data in sorted(by_tech_summary.items()):
        if t_data["recall_at_1"] is not None:
            print(f"  {tech:<14} R@1={t_data['recall_at_1']*100:5.1f}% | R@3={t_data['recall_at_3']*100:5.1f}% | R@5={t_data['recall_at_5']*100:5.1f}% | MRR={t_data['mrr']:.4f} (N={t_data['queries']})")
        else:
            print(f"  {tech:<14} Abstention={t_data['abstention_rate']*100:5.1f}% (N={t_data['queries']})")
    print("--------------------------------------------------------------------------------")
    print(f"E2E Latency (p50):          {latency_p50:.2f} ms")
    print(f"E2E Latency (p95):          {latency_p95:.2f} ms")
    print(f"Embedding Latency (p50):    {embed_p50:.2f} ms")
    print(f"FAISS Search Latency (p50): {search_p50:.2f} ms")
    print(f"Index Vector Count:         {summary['index_statistics']['total_vectors']}")
    print(f"Index Disk Size:            {total_index_bytes / 1024:.1f} KB")
    print("--------------------------------------------------------------------------------")
    print(f"Machine-readable results saved to: {output_file}")
    print("================================================================================")

    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate semantic retrieval quality on DevDocs RAG index.")
    parser.add_argument("--dataset", default="data/eval/retrieval_eval_dataset.json", help="Path to evaluation dataset JSON")
    parser.add_argument("--output", default="data/eval/retrieval_eval_results.json", help="Path to save evaluation results JSON")
    parser.add_argument("--top-k", type=int, default=8, help="Number of chunks to retrieve")
    parser.add_argument("--min-score", type=float, default=None, help="Relevance score threshold")
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        top_k=args.top_k,
        min_score=args.min_score
    )
