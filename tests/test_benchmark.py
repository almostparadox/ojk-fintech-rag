# tests/test_benchmark.py
from pathlib import Path
import json
from unittest.mock import MagicMock
import pytest
from src.evaluation.benchmark import run_evaluation_benchmark, BenchmarkEvaluator
from src.ingestion.indexer import LegalIndexer
from src.config import LegalChunk

def test_golden_dataset_structure():
    dataset_path = Path(__file__).parent.parent / "src" / "evaluation" / "golden_dataset.json"
    assert dataset_path.exists(), "golden_dataset.json must exist"
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, list)
    assert len(data) >= 15, f"Expected at least 15 questions, found {len(data)}"
    
    for item in data:
        assert "id" in item
        assert "query" in item
        assert "expected_status" in item
        assert "is_out_of_scope" in item
        if not item["is_out_of_scope"]:
            assert "expected_reg" in item
            assert "expected_pasal" in item

def test_benchmark_evaluator_scoring(tmp_path):
    storage_dir = tmp_path / "storage"
    
    # Setup index with sample data
    indexer = LegalIndexer(storage_dir=storage_dir)
    chunks = [
        LegalChunk(
            id="POJK_10_2022_PASAL_8",
            reg_id="POJK 10/POJK.05/2022",
            reg_title="LPBBTI",
            status="Berlaku",
            bab="BAB III",
            pasal="Pasal 8",
            legal_ref="POJK 10/POJK.05/2022 Pasal 8",
            content="Modal disetor minimal Rp25 miliar."
        ),
        LegalChunk(
            id="POJK_77_2016_PASAL_4",
            reg_id="POJK 77/POJK.01/2016",
            reg_title="LPMUBTI Lama",
            status="Dicabut",
            bab="BAB II",
            pasal="Pasal 4",
            legal_ref="POJK 77/POJK.01/2016 Pasal 4",
            content="Modal disetor minimal Rp1 miliar."
        )
    ]
    indexer.index_chunks(chunks)
    
    evaluator = BenchmarkEvaluator(storage_dir=storage_dir)
    results = evaluator.evaluate()
    
    assert "total_queries" in results
    assert "retrieval_recall_at_k" in results
    assert "status_accuracy" in results
    assert "refusal_accuracy" in results
    assert results["total_queries"] >= 15
    assert results["retrieval_recall_at_k"] >= 0.0
    assert results["recall_hits"] >= 1

def test_run_evaluation_benchmark_returns_report(tmp_path):
    storage_dir = tmp_path / "storage"
    indexer = LegalIndexer(storage_dir=storage_dir)
    chunks = [
        LegalChunk(
            id="POJK_10_2022_PASAL_8",
            reg_id="POJK 10/POJK.05/2022",
            reg_title="LPBBTI",
            status="Berlaku",
            bab="BAB III",
            pasal="Pasal 8",
            legal_ref="POJK 10/POJK.05/2022 Pasal 8",
            content="Modal disetor minimal Rp25 miliar."
        )
    ]
    indexer.index_chunks(chunks)

    report = run_evaluation_benchmark(storage_dir=storage_dir)
    assert isinstance(report, str)
    assert "BENCHMARK EVALUATION REPORT" in report
    assert "Retrieval Recall" in report

def test_benchmark_with_mock_generator(tmp_path):
    storage_dir = tmp_path / "storage_llm"
    indexer = LegalIndexer(storage_dir=storage_dir)
    chunks = [
        LegalChunk(
            id="POJK_10_2022_PASAL_8",
            reg_id="POJK 10/POJK.05/2022",
            reg_title="LPBBTI",
            status="Berlaku",
            bab="BAB III",
            pasal="Pasal 8",
            legal_ref="POJK 10/POJK.05/2022 Pasal 8",
            content="Modal disetor minimal Rp25 miliar."
        )
    ]
    indexer.index_chunks(chunks)

    mock_generator = MagicMock()
    # Return compliant citation for regular questions, and refusal for recipe/traffic
    def mock_generate(query, contexts):
        if "rendang" in query.lower() or "kecepatan" in query.lower():
            return "Berdasarkan peraturan yang tersedia dalam basis data, dasar hukum untuk pertanyaan ini tidak ditemukan."
        return "Berdasarkan POJK 10/POJK.05/2022 Pasal 8, modal disetor adalah Rp25 miliar."

    mock_generator.generate_response.side_effect = mock_generate

    evaluator = BenchmarkEvaluator(storage_dir=storage_dir, generator=mock_generator)
    results = evaluator.evaluate(with_llm=True)
    assert results["citation_accuracy"] is not None
    assert results["refusal_accuracy"] == 1.0
    report = evaluator.format_report(results)
    assert "Citation Precision" in report
