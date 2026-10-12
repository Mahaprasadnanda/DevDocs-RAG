import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from scripts.evaluate_retrieval import calculate_percentile, run_evaluation
from app.retrieval.models import RetrievalResult


def test_calculate_percentile_empty():
    assert calculate_percentile([], 50) == 0.0
    assert calculate_percentile([], 95) == 0.0


def test_calculate_percentile_single_and_multiple():
    # Single value
    assert calculate_percentile([10.0], 50) == 10.0
    assert calculate_percentile([10.0], 95) == 10.0

    # Multiple values: 1, 2, 3, 4, 5
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert calculate_percentile(vals, 0) == 1.0
    assert calculate_percentile(vals, 50) == 3.0
    assert calculate_percentile(vals, 100) == 5.0


def test_run_evaluation_synthetic_dataset(tmp_path):
    synthetic_dataset = [
        {
            "id": "syn-01",
            "query": "How to create path parameters?",
            "category": "api_usage",
            "answerable": True,
            "expected_document_ids": ["doc-path-params"],
            "expected_urls": ["https://fastapi.tiangolo.com/tutorial/path-params/"]
        },
        {
            "id": "syn-02",
            "query": "How to raise HTTP exception?",
            "category": "api_usage",
            "answerable": True,
            "expected_document_ids": ["doc-errors"],
            "expected_urls": ["https://fastapi.tiangolo.com/tutorial/handling-errors/"]
        },
        {
            "id": "syn-ood-01",
            "query": "What is Quantum Chromodynamics?",
            "category": "out_of_domain",
            "answerable": False,
            "expected_document_ids": [],
            "expected_urls": []
        }
    ]

    dataset_path = tmp_path / "synthetic_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_dataset, f)

    output_path = tmp_path / "synthetic_results.json"

    # Mock retrieval service
    mock_retrieval_service = MagicMock()
    mock_retrieval_service.embedder.encode.return_value = [[0.1] * 384]
    mock_retrieval_service.store.index.ntotal = 10
    mock_retrieval_service.store.index.search.return_value = ([[0.9]], [[0]])
    mock_retrieval_service.store.manifest_path = str(tmp_path / "manifest.json")
    mock_retrieval_service.store.metadata = []

    # Manifest file
    with open(mock_retrieval_service.store.manifest_path, "w", encoding="utf-8") as f:
        json.dump({"dimension": 384, "vector_count": 10}, f)

    # Return different results based on query
    def mock_retrieve(query, top_k=8, min_score=None):
        if "Quantum" in query:
            return []  # Abstained
        elif "path parameters" in query:
            return [
                RetrievalResult(
                    chunk_id="chunk-1",
                    text="Path params text 1",
                    source_url="https://fastapi.tiangolo.com/tutorial/path-params/",
                    document_id="doc-path-params",
                    section_title="Path Params",
                    score=0.88
                ),
                RetrievalResult(
                    chunk_id="chunk-2",
                    text="Path params text 2",
                    source_url="https://fastapi.tiangolo.com/tutorial/path-params/",
                    document_id="doc-path-params",
                    section_title="Path Params Details",
                    score=0.85
                )
            ]
        elif "HTTP exception" in query:
            return [
                RetrievalResult(
                    chunk_id="chunk-other",
                    text="Other page text",
                    source_url="https://fastapi.tiangolo.com/tutorial/first-steps/",
                    document_id="doc-first-steps",
                    section_title="First Steps",
                    score=0.80
                ),
                RetrievalResult(
                    chunk_id="chunk-3",
                    text="Errors text",
                    source_url="https://fastapi.tiangolo.com/tutorial/handling-errors/",
                    document_id="doc-errors",
                    section_title="Errors",
                    score=0.75
                )
            ]
        return []

    mock_retrieval_service.retrieve.side_effect = mock_retrieve

    with patch("scripts.evaluate_retrieval.retrieval_service", mock_retrieval_service):
        results = run_evaluation(
            dataset_path=str(dataset_path),
            output_path=str(output_path),
            top_k=5,
            min_score=0.3
        )

    summary = results["summary"]
    assert summary["dataset_size"] == 3
    assert summary["in_domain_queries"] == 2
    assert summary["out_of_domain_queries"] == 1

    # In syn-01: rank 1 match (doc-path-params)
    # In syn-02: rank 2 match (first was doc-first-steps, second was doc-errors)
    # Recall@1: 1 / 2 = 0.5
    # Recall@3: 2 / 2 = 1.0
    # Recall@5: 2 / 2 = 1.0
    # MRR: (1/1 + 1/2) / 2 = 1.5 / 2 = 0.75
    # Abstention rate: 1 / 1 = 1.0
    quality = summary["retrieval_quality"]
    assert quality["recall_at_1"] == 0.5
    assert quality["recall_at_3"] == 1.0
    assert quality["recall_at_5"] == 1.0
    assert quality["mean_reciprocal_rank_mrr"] == 0.75

    abstention = summary["abstention_quality"]
    assert abstention["abstention_rate"] == 1.0
    assert abstention["out_of_domain_abstained"] == 1

    assert output_path.exists()
