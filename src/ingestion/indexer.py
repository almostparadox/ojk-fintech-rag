import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import lancedb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.config import LegalChunk, settings

class LegalIndexer:
    def __init__(self, storage_dir: Optional[Path] = None, embedding_model_name: Optional[str] = None):
        self.storage_dir = Path(storage_dir or settings.STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model_name or settings.EMBEDDING_MODEL
        self._embedder = None
        self._lance_db = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @property
    def lance_db(self):
        if self._lance_db is None:
            lance_path = self.storage_dir / "lancedb"
            self._lance_db = lancedb.connect(str(lance_path))
        return self._lance_db

    def index_chunks(self, chunks: List[LegalChunk], append: bool = False) -> None:
        if not chunks:
            return

        combined_chunks = chunks
        if append:
            bm25_path = self.storage_dir / "bm25.pkl"
            if bm25_path.exists():
                try:
                    with open(bm25_path, "rb") as f:
                        existing_data = pickle.load(f)
                    existing_raw = existing_data.get("chunks", [])
                    chunk_map: Dict[str, LegalChunk] = {
                        c["id"]: LegalChunk(**c) for c in existing_raw
                    }
                    for c in chunks:
                        chunk_map[c.id] = c
                    combined_chunks = list(chunk_map.values())
                except Exception:
                    combined_chunks = chunks

        # 1. BM25 Tokenization & Indexing
        tokenized_corpus = [
            c.to_search_text().lower().split() for c in combined_chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_data = {
            "bm25": bm25,
            "chunks": [c.model_dump() for c in combined_chunks]
        }
        with open(self.storage_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_data, f)

        # 2. Vector Embedding & LanceDB Indexing
        texts = [c.to_search_text() for c in combined_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        data = []
        for c, emb in zip(combined_chunks, embeddings):
            row = c.model_dump()
            row["vector"] = emb
            data.append(row)

        table_name = "legal_chunks"
        self.lance_db.create_table(table_name, data=data, mode="overwrite")
