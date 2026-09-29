# OJK & Fintech Indonesian Legal RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade, structure-aware Indonesian legal RAG system for OJK/Fintech regulations and UU PDP featuring hybrid retrieval, temporal validity filtering, strict statutory citation guardrails, CLI, and Streamlit UI.

**Architecture:** Custom regex state-machine parser segments Indonesian regulatory PDFs by `BAB` -> `Pasal` -> `Ayat` -> `Huruf`. Storage layer fuses LanceDB dense embeddings with BM25 sparse keyword rankings using Reciprocal Rank Fusion (RRF) and validity pre-filtering. Generation connects via 9router's OpenAI-compatible endpoint with refusal and citation guardrails.

**Tech Stack:** Python 3.11+ (uv managed), LanceDB, sentence-transformers, rank-bm25, PyMuPDF/pypdf, OpenAI SDK, Streamlit, Typer/argparse, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-ojk-fintech-rag-design.md`

## Global Constraints

- **Python Version:** 3.11+ managed by `uv`.
- **Primary Data Format:** Structured `LegalChunk` with explicit `reg_id`, `reg_title`, `status`, `bab`, `pasal`, `legal_ref`, and `content`.
- **Validity Awareness:** Revoked regulations (`status == "Dicabut"`) must be filterable by default to prevent legal hallucination.
- **Dependency Simplicity:** Zero required background Docker daemons; LanceDB and BM25 run embedded locally.
- **Portability:** Sample datasets pre-bundled so any user can run `python cli.py bootstrap` and query immediately with only an API key.

---

### Task 1: Environment & Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `tests/__init__.py`
- Create: `tests/test_scaffold.py`

**Interfaces:**
- Produces: Project dependencies file and directory structure for `src/`, `data/`, `storage/`, `tests/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffold.py
import os
from pathlib import Path

def test_project_structure_exists():
    root = Path(__file__).parent.parent
    expected_dirs = [
        root / "src",
        root / "data" / "raw",
        root / "data" / "processed",
        root / "data" / "sample",
        root / "storage",
        root / "tests",
    ]
    for d in expected_dirs:
        assert d.exists() and d.is_dir(), f"Missing directory: {d}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_scaffold.py -v`
Expected: FAIL (missing directories or pytest not installed).

- [ ] **Step 3: Create directory structure and configuration files**

Create directories:
```bash
mkdir -p src/ingestion src/retrieval src/generation src/evaluation data/raw data/processed data/sample storage tests
touch src/__init__.py src/ingestion/__init__.py src/retrieval/__init__.py src/generation/__init__.py src/evaluation/__init__.py tests/__init__.py
```

Create `pyproject.toml`:
```toml
[project]
name = "ojk-fintech-rag"
version = "0.1.0"
description = "Indonesian Legal & Fintech Regulatory RAG (OJK, SEOJK, UU PDP)"
authors = [{ name = "almostparadox" }]
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "lancedb>=0.6.0",
    "sentence-transformers>=3.0.0",
    "rank-bm25>=0.2.2",
    "pypdf>=4.0.0",
    "openai>=1.30.0",
    "pydantic>=2.7.0",
    "python-dotenv>=1.0.0",
    "rich>=13.7.0",
    "streamlit>=1.35.0",
    "pytest>=8.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Create `requirements.txt`:
```text
lancedb>=0.6.0
sentence-transformers>=3.0.0
rank-bm25>=0.2.2
pypdf>=4.0.0
openai>=1.30.0
pydantic>=2.7.0
python-dotenv>=1.0.0
rich>=13.7.0
streamlit>=1.35.0
pytest>=8.0.0
```

Create `.env.example`:
```env
NINEROUTER_API_KEY=your_9router_key_here
NINEROUTER_BASE_URL=https://api.9router.com/v1
DEFAULT_MODEL=deepseek-chat
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
STORAGE_DIR=./storage
```

Create `.gitignore`:
```gitignore
__pycache__/
*.py[cod]
*$py.class
.venv/
storage/
.env
.DS_Store
*.log
.pytest_cache/
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_scaffold.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt .env.example .gitignore tests/ src/ data/
git commit -m "chore: scaffold project structure and configuration"
```

---

### Task 2: Core Configuration and Data Models

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` class loading environment variables, and `LegalChunk` Pydantic model representing each legal pasal chunk.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from src.config import Settings, LegalChunk

def test_settings_default_values():
    settings = Settings(NINEROUTER_API_KEY="test-key")
    assert settings.NINEROUTER_API_KEY == "test-key"
    assert settings.NINEROUTER_BASE_URL == "https://api.9router.com/v1"
    assert settings.DEFAULT_MODEL == "deepseek-chat"

def test_legal_chunk_model():
    chunk = LegalChunk(
        id="POJK_10_2022_PASAL_8",
        reg_id="POJK 10/POJK.05/2022",
        reg_title="Layanan Pendanaan Bersama Berbasis Teknologi Informasi",
        status="Berlaku",
        bab="BAB III PERIZINAN",
        pasal="Pasal 8",
        legal_ref="POJK 10/POJK.05/2022 Pasal 8",
        content="Pasal 8\n(1) Modal disetor minimal Rp25 miliar.",
        metadata={"tahun": 2022}
    )
    assert chunk.status == "Berlaku"
    assert "Pasal 8" in chunk.content
    assert chunk.to_search_text().startswith("[POJK 10/POJK.05/2022]")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.config'`).

- [ ] **Step 3: Write minimal implementation in `src/config.py`**

```python
# src/config.py
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    NINEROUTER_API_KEY: str = Field(default_factory=lambda: os.getenv("NINEROUTER_API_KEY", ""))
    NINEROUTER_BASE_URL: str = Field(default_factory=lambda: os.getenv("NINEROUTER_BASE_URL", "https://api.9router.com/v1"))
    DEFAULT_MODEL: str = Field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "deepseek-chat"))
    EMBEDDING_MODEL: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    STORAGE_DIR: Path = Field(default_factory=lambda: Path(os.getenv("STORAGE_DIR", "./storage")).resolve())
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")

class LegalChunk(BaseModel):
    id: str
    reg_id: str
    reg_title: str
    status: str = "Berlaku"  # "Berlaku", "Diubah", "Dicabut"
    bab: str = ""
    pasal: str
    legal_ref: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_search_text(self) -> str:
        status_tag = f"[{self.status.upper()}]"
        return f"[{self.reg_id}] {status_tag} {self.bab} - {self.pasal}\n{self.content}"

settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add settings and LegalChunk data model"
```

---

### Task 3: Legal Structure-Aware Hierarchy Parser

**Files:**
- Create: `src/ingestion/legal_parser.py`
- Test: `tests/test_legal_parser.py`

**Interfaces:**
- Consumes: Raw text string of Indonesian regulation.
- Produces: List of `LegalChunk` objects parsed by `BAB` and `Pasal`, preserving all `Ayat` and `Huruf`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_legal_parser.py
from src.ingestion.legal_parser import LegalParser

SAMPLE_LEGAL_TEXT = """
PERATURAN OTORITAS JASA KEUANGAN
NOMOR 10 /POJK.05/2022
TENTANG
LAYANAN PENDANAAN BERSAMA BERBASIS TEKNOLOGI INFORMASI

BAB I
KETENTUAN UMUM
Pasal 1
Dalam Peraturan Otoritas Jasa Keuangan ini yang dimaksud dengan:
1. Layanan Pendanaan Bersama Berbasis Teknologi Informasi...

BAB III
PERIZINAN DAN KELEMBAGAAN
Pasal 8
(1) Penyelenggara harus memiliki modal disetor pada saat pendirian paling sedikit Rp25.000.000.000,00 (dua puluh lima miliar rupiah).
(2) Modal disetor sebagaimana dimaksud pada ayat (1) wajib disetor secara tunai.

Pasal 9
Penyelenggara dilarang melakukan perubahan kepemilikan saham tanpa persetujuan OJK.
"""

def test_parse_sample_legal_text():
    parser = LegalParser()
    chunks = parser.parse_text(
        text=SAMPLE_LEGAL_TEXT,
        reg_id="POJK 10/POJK.05/2022",
        reg_title="Layanan Pendanaan Bersama Berbasis Teknologi Informasi",
        status="Berlaku"
    )
    assert len(chunks) == 3
    
    # Check Pasal 8
    p8 = next(c for c in chunks if c.pasal == "Pasal 8")
    assert p8.bab == "BAB III PERIZINAN DAN KELEMBAGAAN"
    assert "Rp25.000.000.000,00" in p8.content
    assert "(2) Modal disetor" in p8.content
    assert p8.legal_ref == "POJK 10/POJK.05/2022 Pasal 8"
    assert p8.status == "Berlaku"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_legal_parser.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.ingestion.legal_parser'`).

- [ ] **Step 3: Write implementation in `src/ingestion/legal_parser.py`**

```python
# src/ingestion/legal_parser.py
import re
from typing import List, Optional
from src.config import LegalChunk

class LegalParser:
    BAB_PATTERN = re.compile(r'^(BAB\s+[IVXLCDM]+(?:\s+[^\n]+)?)', re.MULTILINE | re.IGNORECASE)
    PASAL_PATTERN = re.compile(r'^(Pasal\s+\d+)', re.MULTILINE | re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        # Strip running page artifacts and administrative footnotes
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r'^(?:-\s*\d+\s*-|\d+\s+dari\s+\d+|Salinan sesuai dengan aslinya)$', stripped, re.IGNORECASE):
                continue
            lines.append(line)
        return "\n".join(lines)

    def parse_text(
        self,
        text: str,
        reg_id: str,
        reg_title: str,
        status: str = "Berlaku"
    ) -> List[LegalChunk]:
        cleaned = self.clean_text(text)
        chunks: List[LegalChunk] = []

        # Find all BAB headings with positions
        bab_matches = list(self.BAB_PATTERN.finditer(cleaned))
        # Find all Pasal headings with positions
        pasal_matches = list(self.PASAL_PATTERN.finditer(cleaned))

        if not pasal_matches:
            # Fallback if no explicit Pasal found: single chunk
            chunks.append(LegalChunk(
                id=f"{reg_id.replace('/', '_').replace(' ', '_')}_ALL",
                reg_id=reg_id,
                reg_title=reg_title,
                status=status,
                bab="",
                pasal="Semua",
                legal_ref=f"{reg_id} Dokumen Lengkap",
                content=cleaned[:4000]
            ))
            return chunks

        def get_bab_for_position(pos: int) -> str:
            current_bab = ""
            for bm in bab_matches:
                if bm.start() <= pos:
                    current_bab = bm.group(1).strip()
                else:
                    break
            return current_bab

        for i, pm in enumerate(pasal_matches):
            pasal_title = pm.group(1).strip()
            start_idx = pm.start()
            end_idx = pasal_matches[i + 1].start() if i + 1 < len(pasal_matches) else len(cleaned)

            pasal_body = cleaned[start_idx:end_idx].strip()
            bab_title = get_bab_for_position(start_idx)
            
            clean_reg_id = re.sub(r'[^a-zA-Z0-9]', '_', reg_id)
            clean_pasal = re.sub(r'[^a-zA-Z0-9]', '_', pasal_title)
            chunk_id = f"{clean_reg_id}_{clean_pasal}"

            chunks.append(LegalChunk(
                id=chunk_id,
                reg_id=reg_id,
                reg_title=reg_title,
                status=status,
                bab=bab_title,
                pasal=pasal_title,
                legal_ref=f"{reg_id} {pasal_title}",
                content=pasal_body
            ))

        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_legal_parser.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/legal_parser.py tests/test_legal_parser.py
git commit -m "feat: implement structure-aware legal hierarchy parser"
```

---

### Task 4: Pre-bundled Sample Regulations Data

**Files:**
- Create: `data/sample/pojk_10_2022.json`
- Create: `data/sample/seojk_19_2023.json`
- Create: `data/sample/uu_27_2022.json`
- Create: `data/sample/pojk_77_2016_revoked.json`
- Test: `tests/test_sample_data.py`

**Interfaces:**
- Produces: JSON datasets of real Indonesian fintech regulations ready for zero-latency indexing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sample_data.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_sample_data.py -v`
Expected: FAIL (missing sample files).

- [ ] **Step 3: Create pre-bundled JSON datasets**

Create `data/sample/pojk_10_2022.json`:
Contains active LPBBTI rules (Pasal 8 on Rp25B capital, Pasal 26 on Rp2B max funding limit, Pasal 29 on risk mitigation). Status: `Berlaku`.

Create `data/sample/seojk_19_2023.json`:
Contains economic benefit / interest rate caps (0.3% / 0.1% per day, collection ethics). Status: `Berlaku`.

Create `data/sample/uu_27_2022.json`:
Contains Personal Data Protection rules (Pasal 20 legal basis for processing, Pasal 57 administrative sanctions). Status: `Berlaku`.

Create `data/sample/pojk_77_2016_revoked.json`:
Contains old 2016 P2P rules (Pasal 4 Rp1B capital). Status: `Dicabut` (revoked by POJK 10/2022).

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_sample_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data/sample/ tests/test_sample_data.py
git commit -m "feat: add curated sample Indonesian fintech regulations data"
```

---

### Task 5: PDF and File Ingestor

**Files:**
- Create: `src/ingestion/pdf_loader.py`
- Test: `tests/test_pdf_loader.py`

**Interfaces:**
- Consumes: PDF file path or text file path.
- Produces: Cleaned string representation.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pdf_loader.py
from pathlib import Path
from src.ingestion.pdf_loader import load_document

def test_load_text_document(tmp_path):
    doc_path = tmp_path / "sample_rule.txt"
    doc_path.write_text("Pasal 1\nKetentuan umum mengenai fintech.", encoding="utf-8")
    
    text = load_document(doc_path)
    assert "Pasal 1" in text
    assert "fintech" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_pdf_loader.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.ingestion.pdf_loader'`).

- [ ] **Step 3: Write implementation in `src/ingestion/pdf_loader.py`**

```python
# src/ingestion/pdf_loader.py
from pathlib import Path
import pypdf

def load_document(file_path: Path) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        return "\n".join(text_parts)
    else:
        return path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_pdf_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pdf_loader.py tests/test_pdf_loader.py
git commit -m "feat: implement document loader with pdf and text support"
```

---

### Task 6: LanceDB & BM25 Inverted Indexer

**Files:**
- Create: `src/ingestion/indexer.py`
- Test: `tests/test_indexer.py`

**Interfaces:**
- Consumes: List of `LegalChunk`.
- Produces: Saved LanceDB vector table and BM25 index on disk in `storage/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_indexer.py
import shutil
from pathlib import Path
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_indexer.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.ingestion.indexer'`).

- [ ] **Step 3: Write implementation in `src/ingestion/indexer.py`**

```python
# src/ingestion/indexer.py
import pickle
from pathlib import Path
from typing import List, Dict, Any
import lancedb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.config import LegalChunk, settings

class LegalIndexer:
    def __init__(self, storage_dir: Path = None, embedding_model_name: str = None):
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

    def index_chunks(self, chunks: List[LegalChunk]) -> None:
        if not chunks:
            return

        # 1. BM25 Tokenization & Indexing
        tokenized_corpus = [
            c.to_search_text().lower().split() for c in chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_data = {
            "bm25": bm25,
            "chunks": [c.model_dump() for c in chunks]
        }
        with open(self.storage_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_data, f)

        # 2. Vector Embedding & LanceDB Indexing
        texts = [c.to_search_text() for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        data = []
        for c, emb in zip(chunks, embeddings):
            row = c.model_dump()
            row["vector"] = emb
            data.append(row)

        table_name = "legal_chunks"
        if table_name in self.lance_db.table_names():
            self.lance_db.drop_table(table_name)
        self.lance_db.create_table(table_name, data=data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_indexer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/indexer.py tests/test_indexer.py
git commit -m "feat: implement LanceDB vector and BM25 inverted indexer"
```

---

### Task 7: Hybrid Search Engine and Temporal Status Filter

**Files:**
- Create: `src/retrieval/status_filter.py`
- Create: `src/retrieval/hybrid_search.py`
- Test: `tests/test_hybrid_search.py`

**Interfaces:**
- Consumes: Natural language query string, `active_only` boolean flag, and top-k parameter.
- Produces: List of ranked `LegalChunk` with computed RRF relevance score and validity metadata.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hybrid_search.py
from pathlib import Path
from src.config import LegalChunk
from src.ingestion.indexer import LegalIndexer
from src.retrieval.hybrid_search import HybridSearcher

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
    
    # Exact Pasal lookup should score highest
    pasal_results = searcher.search(query="Pasal 8 POJK 10/2022", top_k=5, active_only=True)
    assert pasal_results[0].chunk.pasal == "Pasal 8"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_hybrid_search.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.retrieval.hybrid_search'`).

- [ ] **Step 3: Write implementation in `src/retrieval/status_filter.py` and `src/retrieval/hybrid_search.py`**

```python
# src/retrieval/status_filter.py
from typing import List
from src.config import LegalChunk

def filter_by_status(chunks: List[LegalChunk], active_only: bool = True) -> List[LegalChunk]:
    if not active_only:
        return chunks
    return [c for c in chunks if c.status.strip().lower() != "dicabut"]
```

```python
# src/retrieval/hybrid_search.py
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
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
    def __init__(self, storage_dir: Path = None):
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
        # 1. BM25 Search
        bm25_obj = self.bm25_data["bm25"]
        all_chunks = [LegalChunk(**c) for c in self.bm25_data["chunks"]]
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

        # Sort combined candidates
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_hybrid_search.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/status_filter.py src/retrieval/hybrid_search.py tests/test_hybrid_search.py
git commit -m "feat: implement hybrid search with RRF and legal validity filtering"
```

---

### Task 8: 9router Client & Grounded Legal Prompting

**Files:**
- Create: `src/generation/prompt.py`
- Create: `src/generation/client.py`
- Test: `tests/test_generation.py`

**Interfaces:**
- Consumes: Query string, list of `LegalChunk` search results.
- Produces: LLM streaming / completed answer formatted with statutory citations or grounded refusal.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generation.py
from src.config import LegalChunk
from src.generation.prompt import build_system_prompt, build_user_prompt
from src.generation.client import LegalGenerator

def test_prompt_formatting():
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
    sys_prompt = build_system_prompt()
    user_prompt = build_user_prompt(query="Berapa modal fintech?", contexts=chunks)
    
    assert "POJK 10/POJK.05/2022 Pasal 8" in user_prompt
    assert "Rp25 miliar" in user_prompt
    assert "Berlaku" in user_prompt
    assert "dasar hukum" in sys_prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_generation.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.generation.prompt'`).

- [ ] **Step 3: Write implementation in `src/generation/prompt.py` and `src/generation/client.py`**

```python
# src/generation/prompt.py
from typing import List
from src.config import LegalChunk

def build_system_prompt() -> str:
    return """Anda adalah Asisten Regulasi & Kepatuhan Hukum Fintech Indonesia (OJK, BI, dan UU PDP).
Tugas Anda adalah menjawab pertanyaan pengguna secara akurat HANYA berdasarkan konteks peraturan perundang-undangan yang diberikan.

ATURAN KETAT:
1. Setiap pernyataan atau angka wajib mencantumkan rujukan pasal eksplisit dalam format: [Nama Peraturan, Pasal X ayat Y].
2. Cantumkan status keberlakuan peraturan yang dirujuk (misal: [STATUS: BERLAKU] atau [STATUS: DICABUT/DIUBAH]).
3. DILARANG KERAS berhalusinasi atau memberikan interpretasi di luar teks hukum yang tersedia.
4. Jika konteks yang diberikan TIDAK memuat dasar hukum untuk menjawab pertanyaan, jawab dengan tegas:
   "Berdasarkan peraturan yang tersedia dalam basis data, dasar hukum untuk pertanyaan ini tidak ditemukan."
5. Jawaban harus profesional, terstruktur, dan objektif."""

def build_user_prompt(query: str, contexts: List[LegalChunk]) -> str:
    context_blocks = []
    for i, c in enumerate(contexts, 1):
        context_blocks.append(
            f"--- KONTEKS {i} ---\n"
            f"Peraturan: {c.reg_id} ({c.reg_title})\n"
            f"Status: {c.status}\n"
            f"Bagian: {c.bab} - {c.pasal}\n"
            f"Rujukan: {c.legal_ref}\n"
            f"Isi Ketentuan:\n{c.content}\n"
        )
    joined_contexts = "\n".join(context_blocks)
    return f"""Berikut adalah dokumen peraturan yang relevan:

{joined_contexts}

PERTANYAAN PENGGUNA:
{query}

JAWABAN (dengan rujukan pasal dan status):"""
```

```python
# src/generation/client.py
from typing import Generator, List
from openai import OpenAI
from src.config import LegalChunk, settings
from src.generation.prompt import build_system_prompt, build_user_prompt

class LegalGenerator:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or settings.NINEROUTER_API_KEY
        self.base_url = base_url or settings.NINEROUTER_BASE_URL
        self.model = model or settings.DEFAULT_MODEL
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key or "dummy_key",
                base_url=self.base_url
            )
        return self._client

    def generate_response(self, query: str, contexts: List[LegalChunk]) -> str:
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(query, contexts)}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content or ""

    def stream_response(self, query: str, contexts: List[LegalChunk]) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(query, contexts)}
        ]
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_generation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/generation/prompt.py src/generation/client.py tests/test_generation.py
git commit -m "feat: implement 9router LLM client and legal citation prompt guardrails"
```

---

### Task 9: Evaluation Suite and Benchmark Harness

**Files:**
- Create: `src/evaluation/golden_dataset.json`
- Create: `src/evaluation/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: Golden dataset of Indonesian legal questions with ground-truth citations.
- Produces: Evaluation report measuring Citation Recall, Status Accuracy, and Refusal Groundedness.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark.py
from pathlib import Path
from src.evaluation.benchmark import run_evaluation_benchmark

def test_evaluation_benchmark_runner(tmp_path):
    # Test benchmark scoring logic on sample golden dataset
    dataset_path = Path(__file__).parent.parent / "src" / "evaluation" / "golden_dataset.json"
    assert dataset_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_benchmark.py -v`
Expected: FAIL (missing files).

- [ ] **Step 3: Create golden dataset & benchmark runner**

Create `src/evaluation/golden_dataset.json` with 15 curated questions:
1. Minimum capital requirements (POJK 10/2022 Pasal 8).
2. Maximum funding limit per borrower (POJK 10/2022 Pasal 26).
3. Economic benefit / daily interest rate cap (SEOJK 19/2023).
4. Collection rules and ethical hours (SEOJK 19/2023).
5. Legal grounds for personal data processing (UU 27/2022 Pasal 20).
6. Sanctions for data breach under UU PDP (UU 27/2022 Pasal 57).
7. Old POJK 77/2016 revocation check (must NOT be cited for active rules).
8. Out-of-scope negative question (e.g. traffic rules or crypto spot margin) -> must trigger refusal.
... up to 15 questions.

Create `src/evaluation/benchmark.py` calculating:
- Retrieval Recall@K (did retrieved chunks contain the expected `target_pasal`?)
- Citation Precision (did generated answer cite the expected regulation and article?)
- Status Check (did it filter or flag revoked status?)

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/ tests/test_benchmark.py
git commit -m "feat: implement legal benchmark harness and golden dataset"
```

---

### Task 10: CLI Application & One-Command Bootstrap

**Files:**
- Create: `cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: Command line executable with subcommands `bootstrap`, `query`, `ingest`, and `eval`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import subprocess
import sys

def test_cli_help():
    result = subprocess.run([sys.executable, "cli.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "bootstrap" in result.stdout
    assert "query" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_cli.py -v`
Expected: FAIL (`cli.py` does not exist).

- [ ] **Step 3: Implement `cli.py`**

```python
# cli.py
import argparse
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import LegalChunk, settings
from src.ingestion.indexer import LegalIndexer
from src.retrieval.hybrid_search import HybridSearcher
from src.generation.client import LegalGenerator
from src.ingestion.pdf_loader import load_document
from src.ingestion.legal_parser import LegalParser
from src.evaluation.benchmark import run_evaluation_benchmark

console = Console()

def cmd_bootstrap(args):
    console.print("[bold green]🚀 Memulai Bootstrap Data Regulasi OJK & UU PDP...[/bold green]")
    sample_dir = settings.DATA_DIR / "sample"
    all_chunks = []
    for f in sample_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            all_chunks.extend([LegalChunk(**item) for item in data])

    console.print(f"📦 Mengindeks [cyan]{len(all_chunks)}[/cyan] pasal regulasi...")
    indexer = LegalIndexer()
    indexer.index_chunks(all_chunks)
    console.print("[bold green]✓ Bootstrap selesai! Database LanceDB & BM25 siap digunakan.[/bold green]")

def cmd_query(args):
    searcher = HybridSearcher()
    console.print(f"\n[bold yellow]🔍 Mencari dasar hukum untuk:[/bold yellow] [italic]{args.query}[/italic]\n")
    results = searcher.search(args.query, top_k=args.top_k, active_only=not args.include_revoked)

    if not results:
        console.print("[red]Tidak ada konteks hukum yang ditemukan.[/red]")
        return

    table = Table(title="Dasar Hukum Terkait (Top Reranked)")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Peraturan", style="magenta")
    table.add_column("Pasal", style="green")
    table.add_column("Status", style="bold")
    table.add_column("RRF Score", style="yellow")

    for i, r in enumerate(results, 1):
        status_color = "green" if r.chunk.status == "Berlaku" else "red"
        table.add_row(
            str(i),
            r.chunk.reg_id,
            r.chunk.pasal,
            f"[{status_color}]{r.chunk.status}[/{status_color}]",
            f"{r.score:.4f}"
        )
    console.print(table)

    if not args.no_llm:
        console.print("\n[bold cyan]🤖 Analisis Regulasi (9router):[/bold cyan]")
        generator = LegalGenerator()
        contexts = [r.chunk for r in results]
        try:
            for token in generator.stream_response(args.query, contexts):
                sys.stdout.write(token)
                sys.stdout.flush()
            print()
        except Exception as e:
            console.print(f"[red]Gagal memanggil 9router LLM: {e}[/red]")
            console.print("[yellow]Tip: Pastikan NINEROUTER_API_KEY sudah diisi di .env[/yellow]")

def cmd_ingest(args):
    file_path = Path(args.file)
    text = load_document(file_path)
    parser = LegalParser()
    chunks = parser.parse_text(
        text=text,
        reg_id=args.reg_id,
        reg_title=args.reg_title,
        status=args.status
    )
    console.print(f"Parsed {len(chunks)} pasal from {file_path.name}")
    indexer = LegalIndexer()
    indexer.index_chunks(chunks)
    console.print(f"[green]Successfully indexed {file_path.name}[/green]")

def cmd_eval(args):
    console.print("[bold cyan]Menjalankan Benchmark Evaluasi Kepatuhan Hukum...[/bold cyan]")
    report = run_evaluation_benchmark()
    console.print(Panel(report, title="Hasil Benchmark Evaluasi"))

def main():
    parser = argparse.ArgumentParser(description="OJK & Fintech Indonesian Legal RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bootstrap
    sub_boot = subparsers.add_parser("bootstrap", help="Index pre-bundled sample regulations")
    sub_boot.set_defaults(func=cmd_bootstrap)

    # query
    sub_q = subparsers.add_parser("query", help="Ask regulatory question")
    sub_q.add_argument("query", type=str, help="Pertanyaan hukum")
    sub_q.add_argument("--top-k", type=int, default=3, help="Jumlah pasal rujukan")
    sub_q.add_argument("--include-revoked", action="store_true", help="Termasuk aturan yang telah dicabut")
    sub_q.add_argument("--no-llm", action="store_true", help="Tampilkan hasil retrieval saja tanpa LLM")
    sub_q.set_defaults(func=cmd_query)

    # ingest
    sub_ing = subparsers.add_parser("ingest", help="Ingest a new PDF or text regulation")
    sub_ing.add_argument("--file", required=True, help="Path ke file PDF atau teks")
    sub_ing.add_argument("--reg-id", required=True, help="Nomor peraturan (misal POJK 10/POJK.05/2022)")
    sub_ing.add_argument("--reg-title", required=True, help="Tentang peraturan")
    sub_ing.add_argument("--status", default="Berlaku", choices=["Berlaku", "Diubah", "Dicabut"])
    sub_ing.set_defaults(func=cmd_ingest)

    # eval
    sub_eval = subparsers.add_parser("eval", help="Run benchmark suite")
    sub_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cli.py tests/test_cli.py
git commit -m "feat: implement CLI interface with bootstrap, query, ingest, and eval"
```

---

### Task 11: Streamlit Web User Interface

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Produces: Interactive browser application (`app.py`) for live demonstrations.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app.py
from pathlib import Path

def test_streamlit_app_file_valid():
    app_file = Path(__file__).parent.parent / "app.py"
    assert app_file.exists()
    content = app_file.read_text(encoding="utf-8")
    assert "import streamlit as st" in content
    assert "HybridSearcher" in content
    assert "LegalGenerator" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_app.py -v`
Expected: FAIL (`app.py` not found).

- [ ] **Step 3: Implement `app.py`**

```python
# app.py
import streamlit as st
from src.config import settings
from src.retrieval.hybrid_search import HybridSearcher
from src.generation.client import LegalGenerator
from src.ingestion.indexer import LegalIndexer
import json

st.set_page_config(
    page_title="OJK Fintech Regulatory RAG",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ OJK & Fintech Legal Intelligence Assistant")
st.markdown("Asisten Kepatuhan Hukum Fintech Indonesia berbasis **Hybrid RAG** (LanceDB + BM25) dengan validasi status keberlakuan peraturan.")

with st.sidebar:
    st.header("⚙️ Pengaturan & Filter")
    active_only = st.checkbox("Hanya Peraturan Aktif (Berlaku)", value=True, help="Saring otomatis aturan yang telah dicabut (misal POJK 77/2016)")
    top_k = st.slider("Jumlah Rujukan Pasal (Top K)", min_value=1, max_value=8, value=3)
    
    st.divider()
    st.subheader("📚 Status Basis Data")
    if st.button("🔄 Reload / Re-index Sample Data"):
        with st.spinner("Mengindeks data sample..."):
            all_chunks = []
            for f in (settings.DATA_DIR / "sample").glob("*.json"):
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    from src.config import LegalChunk
                    all_chunks.extend([LegalChunk(**item) for item in data])
            LegalIndexer().index_chunks(all_chunks)
            st.success("Basis data berhasil diperbarui!")

    st.markdown("""
    ---
    **Peraturan Terindeks:**
    - `POJK 10/POJK.05/2022` (P2P Lending)
    - `SEOJK 19/SEOJK.06/2023` (Bunga & Etika Penagihan)
    - `UU 27/2022` (Pelindungan Data Pribadi)
    - `POJK 77/POJK.01/2016` (Status: Dicabut)
    """)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "contexts" in msg:
            with st.expander("🔍 Lihat Dasar Hukum yang Dirujuk"):
                for c in msg["contexts"]:
                    st.markdown(f"**{c['legal_ref']}** `[Status: {c['status']}]`")
                    st.text(c['content'])

if prompt := st.chat_input("Tanyakan aturan hukum (contoh: Berapa modal disetor fintech lending?):"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        searcher = HybridSearcher()
        results = searcher.search(prompt, top_k=top_k, active_only=active_only)

        if not results:
            ans = "Berdasarkan peraturan yang tersedia dalam basis data, dasar hukum untuk pertanyaan ini tidak ditemukan."
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
        else:
            with st.expander("🔍 Dasar Hukum Ditemukan (RRF Hybrid Search)", expanded=False):
                for r in results:
                    status_badge = "🟢 BERLAKU" if r.chunk.status == "Berlaku" else "🔴 DICABUT"
                    st.markdown(f"**{r.chunk.legal_ref}** — {status_badge} (Score: `{r.score:.4f}`)")
                    st.caption(f"{r.chunk.bab}")
                    st.code(r.chunk.content, language="text")

            generator = LegalGenerator()
            contexts = [r.chunk for r in results]
            response_container = st.empty()
            full_response = ""
            
            try:
                for chunk in generator.stream_response(prompt, contexts):
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                response_container.markdown(full_response)
            except Exception as e:
                full_response = f"⚠️ Gagal menghubungi LLM: {e}\n\nPastikan `NINEROUTER_API_KEY` terkonfigurasi di `.env`."
                response_container.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "contexts": [r.chunk.model_dump() for r in results]
            })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/test_app.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: implement Streamlit web interface with context inspector"
```

---

### Task 12: Documentation, GitHub Remote, & Friend Quickstart

**Files:**
- Create: `README.md`
- Test: Full end-to-end bootstrap and test run

- [ ] **Step 1: Write `README.md`**

Complete documentation including:
- Problem explanation (why naive RAG fails on Indonesian legal code).
- System architecture diagram.
- 3-minute quickstart for friends:
  1. `git clone ...`
  2. `uv sync` (or `pip install -r requirements.txt`)
  3. `cp .env.example .env` (add 9router key)
  4. `python cli.py bootstrap`
  5. `streamlit run app.py`
- Benchmark evaluation table.

- [ ] **Step 2: Run all unit and integration tests**

Run: `/Users/jks-saddam-dev/.local/bin/uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Run bootstrap CLI check**

Run: `/Users/jks-saddam-dev/.local/bin/uv run python cli.py bootstrap`
Expected: Successfully indexes chunks to `storage/`

- [ ] **Step 4: Commit and setup GitHub repo**

```bash
git add README.md
git commit -m "docs: add comprehensive README with architecture and friend quickstart"
```
Optional: create GitHub private repository via `gh repo create ojk-fintech-rag --private --source=. --remote=origin --push`.
