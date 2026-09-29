from typing import List
from src.config import LegalChunk

def filter_by_status(chunks: List[LegalChunk], active_only: bool = True) -> List[LegalChunk]:
    if not active_only:
        return chunks
    return [c for c in chunks if c.status.strip().lower() != "dicabut"]
