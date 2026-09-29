import json
from pathlib import Path
from src.config import LegalChunk

def test_load_sample_regulations():
    sample_dir = Path(__file__).parent.parent / "data" / "sample"
    sample_files = list(sample_dir.glob("*.json"))
    assert len(sample_files) >= 4, "Should have at least 4 sample regulation datasets"
    
    total_chunks = 0
    statuses = set()
    for f in sample_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            assert isinstance(data, list)
            for item in data:
                chunk = LegalChunk(**item)
                assert chunk.pasal
                assert chunk.content
                statuses.add(chunk.status)
                total_chunks += 1
                
    assert "Berlaku" in statuses
    assert "Dicabut" in statuses
    assert total_chunks >= 10
