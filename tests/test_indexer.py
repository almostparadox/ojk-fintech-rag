import shutil
from pathlib import Path
import pickle
from src.config import LegalChunk
from src.ingestion.indexer import LegalIndexer

def test_indexer_build_and_search(tmp_path):
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
    assert (storage_dir / "bm25.pkl").exists()
    assert (storage_dir / "lancedb").exists()

    # Verify BM25 content
    with open(storage_dir / "bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)
        assert "bm25" in bm25_data
        assert len(bm25_data["chunks"]) == 2

    # Verify LanceDB table content
    table = indexer.lance_db.open_table("legal_chunks")
    assert table.count_rows() == 2
    df = table.to_pandas()
    assert set(df["id"].tolist()) == {"POJK_10_2022_PASAL_8", "POJK_77_2016_PASAL_4"}

def test_indexer_empty_chunks(tmp_path):
    storage_dir = tmp_path / "storage_empty"
    indexer = LegalIndexer(storage_dir=storage_dir)
    indexer.index_chunks([])
    assert not (storage_dir / "bm25.pkl").exists()

def test_indexer_append_mode(tmp_path):
    storage_dir = tmp_path / "storage_append"
    indexer = LegalIndexer(storage_dir=storage_dir)

    initial_chunks = [
        LegalChunk(
            id="POJK_10_2022_PASAL_8",
            reg_id="POJK 10/POJK.05/2022",
            reg_title="LPBBTI",
            status="Berlaku",
            bab="BAB III",
            pasal="Pasal 8",
            legal_ref="POJK 10/POJK.05/2022 Pasal 8",
            content="Modal Rp25 miliar."
        )
    ]
    indexer.index_chunks(initial_chunks, append=False)
    assert indexer.lance_db.open_table("legal_chunks").count_rows() == 1

    # Append new chunk and update existing chunk
    new_chunks = [
        LegalChunk(
            id="POJK_10_2022_PASAL_8",  # duplicate ID: should overwrite
            reg_id="POJK 10/POJK.05/2022",
            reg_title="LPBBTI",
            status="Berlaku",
            bab="BAB III",
            pasal="Pasal 8",
            legal_ref="POJK 10/POJK.05/2022 Pasal 8",
            content="Modal diperbarui Rp25.000.000.000,00."
        ),
        LegalChunk(
            id="POJK_10_2022_PASAL_26",
            reg_id="POJK 10/POJK.05/2022",
            reg_title="LPBBTI",
            status="Berlaku",
            bab="BAB VI",
            pasal="Pasal 26",
            legal_ref="POJK 10/POJK.05/2022 Pasal 26",
            content="Batas maksimum Rp2 miliar."
        )
    ]
    indexer.index_chunks(new_chunks, append=True)

    table = indexer.lance_db.open_table("legal_chunks")
    assert table.count_rows() == 2
    with open(storage_dir / "bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    assert len(bm25_data["chunks"]) == 2
    chunk_map = {c["id"]: c for c in bm25_data["chunks"]}
    assert "diperbarui" in chunk_map["POJK_10_2022_PASAL_8"]["content"]

