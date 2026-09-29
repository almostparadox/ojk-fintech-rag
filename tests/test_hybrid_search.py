import pytest
from pathlib import Path
from src.config import LegalChunk
from src.ingestion.indexer import LegalIndexer
from src.retrieval.status_filter import filter_by_status
from src.retrieval.hybrid_search import HybridSearcher, SearchResult

def test_status_filter_direct():
    chunks = [
        LegalChunk(
            id="C1",
            reg_id="POJK 10/2022",
            reg_title="LPBBTI",
            status="Berlaku",
            pasal="Pasal 1",
            legal_ref="POJK 10/2022 Pasal 1",
            content="Active content"
        ),
        LegalChunk(
            id="C2",
            reg_id="POJK 77/2016",
            reg_title="LPMUBTI",
            status="Dicabut",
            pasal="Pasal 1",
            legal_ref="POJK 77/2016 Pasal 1",
            content="Revoked content"
        )
    ]
    active_chunks = filter_by_status(chunks, active_only=True)
    assert len(active_chunks) == 1
    assert active_chunks[0].id == "C1"

    all_chunks = filter_by_status(chunks, active_only=False)
    assert len(all_chunks) == 2

def test_hybrid_search_active_only(tmp_path):
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
            content="Penyelenggara harus memiliki modal disetor paling sedikit Rp25.000.000.000,00."
        ),
        LegalChunk(
            id="POJK_77_2016_PASAL_4",
            reg_id="POJK 77/POJK.01/2016",
            reg_title="LPMUBTI Lama",
            status="Dicabut",
            bab="BAB II",
            pasal="Pasal 4",
            legal_ref="POJK 77/POJK.01/2016 Pasal 4",
            content="Modal disetor paling sedikit Rp1.000.000.000,00."
        )
    ]
    indexer.index_chunks(chunks)

    searcher = HybridSearcher(storage_dir=storage_dir)

    # Query with active_only=True should exclude revoked POJK 77/2016
    results = searcher.search(query="modal disetor fintech", top_k=5, active_only=True)
    assert len(results) == 1
    assert results[0].chunk.reg_id == "POJK 10/POJK.05/2022"
    assert results[0].chunk.status == "Berlaku"
    assert results[0].score > 0

    # Query with active_only=False should include revoked chunk
    all_results = searcher.search(query="modal disetor fintech", top_k=5, active_only=False)
    assert len(all_results) == 2

    # Exact Pasal lookup should score highest
    pasal_results = searcher.search(query="Pasal 8 POJK 10/2022", top_k=5, active_only=True)
    assert pasal_results[0].chunk.pasal == "Pasal 8"

def test_hybrid_search_missing_index(tmp_path):
    storage_dir = tmp_path / "non_existent_storage"
    searcher = HybridSearcher(storage_dir=storage_dir)
    with pytest.raises(FileNotFoundError):
        searcher.search("test")
