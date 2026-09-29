import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import lancedb
from sentence_transformers import SentenceTransformer
from src.config import LegalChunk, settings
from src.retrieval.status_filter import filter_by_status

class SearchResult(BaseModel):
    chunk: LegalChunk
    score: float
    vector_rank: int = -1
    bm25_rank: int = -1

class HybridSearcher:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir or settings.STORAGE_DIR)
        self._embedder = None
        self._lance_db = None
        self._bm25_data = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self._embedder

    @property
    def lance_table(self):
        if self._lance_db is None:
            self._lance_db = lancedb.connect(str(self.storage_dir / "lancedb"))
        return self._lance_db.open_table("legal_chunks")

    @property
    def bm25_data(self) -> Dict[str, Any]:
        if self._bm25_data is None:
            bm25_path = self.storage_dir / "bm25.pkl"
            if not bm25_path.exists():
                raise FileNotFoundError(f"BM25 index not found at {bm25_path}. Run indexer first.")
            with open(bm25_path, "rb") as f:
                self._bm25_data = pickle.load(f)
        return self._bm25_data

    def search(self, query: str, top_k: int = 5, active_only: bool = True) -> List[SearchResult]:
        all_chunks = [LegalChunk(**c) for c in self.bm25_data.get("chunks", [])]
        if not all_chunks:
            return []

        # 1. BM25 Search
        bm25_obj = self.bm25_data["bm25"]
        query_tokens = query.lower().split()
        bm25_scores = bm25_obj.get_scores(query_tokens)
        
        # Rank BM25
        ranked_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_rank_map: Dict[str, int] = {}
        for rank, idx in enumerate(ranked_bm25_indices):
            if bm25_scores[idx] > 0:
                bm25_rank_map[all_chunks[idx].id] = rank + 1

        # 2. Vector Search
        query_vec = self.embedder.encode(query, show_progress_bar=False).tolist()
        vector_results = self.lance_table.search(query_vec).limit(top_k * 3).to_list()
        vector_rank_map: Dict[str, int] = {}
        chunk_dict: Dict[str, LegalChunk] = {c.id: c for c in all_chunks}
        for rank, row in enumerate(vector_results):
            cid = row["id"]
            vector_rank_map[cid] = rank + 1
            if cid not in chunk_dict:
                row_copy = dict(row)
                row_copy.pop("vector", None)
                row_copy.pop("_distance", None)
                chunk_dict[cid] = LegalChunk(**row_copy)

        # 3. Reciprocal Rank Fusion (RRF)
        candidate_ids = set(bm25_rank_map.keys()) | set(vector_rank_map.keys())
        rrf_scores: Dict[str, float] = {}
        k = 60.0
        for cid in candidate_ids:
            score = 0.0
            if cid in bm25_rank_map:
                score += 1.0 / (k + bm25_rank_map[cid])
            if cid in vector_rank_map:
                score += 1.0 / (k + vector_rank_map[cid])
            rrf_scores[cid] = score

        # Sort combined candidates by RRF score descending
        sorted_ids = sorted(candidate_ids, key=lambda cid: rrf_scores[cid], reverse=True)

        results: List[SearchResult] = []
        for cid in sorted_ids:
            chunk = chunk_dict[cid]
            if active_only and chunk.status.strip().lower() == "dicabut":
                continue
            results.append(SearchResult(
                chunk=chunk,
                score=rrf_scores[cid],
                vector_rank=vector_rank_map.get(cid, -1),
                bm25_rank=bm25_rank_map.get(cid, -1)
            ))
            if len(results) >= top_k:
                break

        return results
