# Design Spec: OJK & Fintech Indonesian Legal RAG (ojk-fintech-rag)

- **Date:** 2026-09-29
- **Status:** Approved
- **Repository:** `/Users/jks-saddam-dev/Private/ojk-fintech-rag`
- **Target Audience:** Portfolio reviewers, fintech compliance engineering hiring managers, peers/friends cloning the repo.

---

## 1. Overview & Business Problem

Standard RAG implementations (chunking by 500 characters + vector similarity) perform poorly on Indonesian legal documents (JDIH, OJK, BI, UU):
1. **Broken legal boundaries:** Arbitrary text chunking breaks *Pasal*, *Ayat*, and *Huruf*, detaching cross-references and conditions.
2. **Temporal validity failure:** Regulations undergo frequent amendment (*Diubah*) or revocation (*Dicabut*). Naive search surfaces revoked regulations without notice, leading to dangerous legal hallucinations.
3. **Exact citation miss:** Pure vector search misses exact article queries (e.g. "Pasal 8 POJK 10/2022") or exact numeric statutory requirements.
4. **Hallucination vulnerability:** Generic prompts summarize or invent legal interpretations without verbatim statutory grounding.

`ojk-fintech-rag` solves these issues with a structure-aware, validity-filtered hybrid retrieval system tailored for Indonesian fintech regulations (POJK/SEOJK) and UU Pelindungan Data Pribadi (PDP).

---

## 2. Architecture & Tech Stack

### 2.1 Core Stack
- **Language:** Python 3.10+
- **Parsing:** PyMuPDF (`fitz`) / `pypdf` + regex legal boundary parser (`BAB` -> `Pasal` -> `Ayat` -> `Huruf`).
- **Dense Vector Index:** LanceDB (serverless, disk-based vector storage embedded in Python).
- **Embeddings:** HuggingFace `sentence-transformers` (`BAAI/bge-m3` or `intfloat/multilingual-e5-small`) running locally on CPU.
- **Sparse Index:** BM25 (`rank-bm25`) for exact keyword and article identifier matching.
- **Fusion:** Reciprocal Rank Fusion (RRF) with configurable weights.
- **LLM Provider:** 9router OpenAI-compatible API (`base_url` + `api_key`).
- **User Interfaces:**
  - Streamlit web interface (`app.py`) for browser inspection and demonstration.
  - Command Line Interface (`cli.py`) for automated queries and ingestion scripting.
- **Evaluation:** Custom legal benchmark harness (`benchmark.py`) measuring citation recall, status awareness, and groundedness.

---

## 3. Directory Structure

```text
ojk-fintech-rag/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── raw/                  # Downloaded raw PDFs (POJK/SEOJK/UU)
│   ├── processed/            # Structured parsed JSON per regulation
│   └── sample/               # Pre-bundled parsed sample regulations for instant testing
├── storage/                  # LanceDB table and BM25 index on disk (gitignored)
├── docs/
│   └── superpowers/specs/    # Architectural specifications
├── src/
│   ├── __init__.py
│   ├── config.py             # App configuration and environment loading
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py     # PDF text extraction and header/footer cleanup
│   │   ├── legal_parser.py   # State-machine chunker respecting legal hierarchy
│   │   └── indexer.py        # Embed and write chunks to LanceDB and BM25
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py  # BM25 + LanceDB vector search with RRF fusion
│   │   └── status_filter.py  # Filter or badge regulations by validity status
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── client.py         # 9router OpenAI client wrapper
│   │   └── prompt.py         # Grounded legal citation prompts
│   └── evaluation/
│       ├── __init__.py
│       ├── benchmark.py      # Automated evaluation runner
│       └── golden_dataset.json # Ground truth Q&A pairs with verified citations
├── app.py                    # Streamlit web application
└── cli.py                    # CLI search and ingestion interface
```

---

## 4. Subsystem Details

### 4.1 Legal Parser (`legal_parser.py`)
- Reads raw text output from `pdf_loader.py`.
- Strips administrative boilerplate (e.g. "Salinan sesuai dengan aslinya", page numbering, signatures).
- Identifies regulation metadata from header:
  - `nomor`: e.g. `10/POJK.05/2022`
  - `tentang`: e.g. `Layanan Pendanaan Bersama Berbasis Teknologi Informasi`
  - `status`: `Berlaku` | `Diubah` | `Dicabut`
- Segmentation rule:
  - Primary chunk unit: **1 Pasal**.
  - All nested `Ayat` and `Huruf` belong to that Pasal chunk to preserve local context.
  - Chunk metadata payload:
    ```json
    {
      "id": "POJK_10_2022_PASAL_8",
      "reg_id": "POJK 10/POJK.05/2022",
      "reg_title": "Layanan Pendanaan Bersama Berbasis Teknologi Informasi",
      "status": "Berlaku",
      "bab": "BAB III PERIZINAN DAN KELEMBAGAAN",
      "pasal": "Pasal 8",
      "legal_ref": "POJK 10/POJK.05/2022 Pasal 8",
      "content": "Pasal 8\n(1) Penyelenggara harus memiliki modal disetor pada saat pendirian paling sedikit Rp25.000.000.000,00 (dua puluh lima miliar rupiah).\n(2) Modal disetor sebagaimana dimaksud pada ayat (1) harus disetor secara tunai..."
    }
    ```

### 4.2 Indexing & Retrieval Engine (`indexer.py`, `hybrid_search.py`)
- **LanceDB Vector Storage:**
  - Chunks embedded into dense vectors using local HuggingFace embeddings (`multilingual-e5-small` or `bge-m3`).
  - Stored in persistent directory `storage/lancedb`.
- **BM25 Inverted Index:**
  - Tokenizes `legal_ref`, `pasal`, `bab`, and `content`.
  - Serialized to `storage/bm25.pkl` for quick reload.
- **Hybrid Fusion (RRF):**
  - Query executed concurrently against Vector and BM25 index.
  - Standard Reciprocal Rank Fusion computes rank scores:
    $$RRF(d) = \frac{1}{60 + r_{vector}(d)} + \frac{1}{60 + r_{bm25}(d)}$$
  - Filter applied: If `active_only=True`, exclude regulations marked `status == "Dicabut"` unless query explicitly requests historical comparison.

### 4.3 Generation & Citation Guardrails (`client.py`, `prompt.py`)
- Client uses `openai.OpenAI` configured with:
  - `base_url = os.getenv("NINEROUTER_BASE_URL", "https://api.9router.com/v1")`
  - `api_key = os.getenv("NINEROUTER_API_KEY")`
  - Default model: `gpt-4o-mini` or `deepseek-chat` configurable in `.env`.
- System prompt rules:
  1. Act as Indonesian FinTech Regulatory Intelligence Assistant.
  2. Answer strictly using the provided context chunks.
  3. Every claim must have an inline citation tag: `[Peraturan, Pasal X ayat Y]`.
  4. Explicitly highlight legal status (e.g. `[STATUS: BERLAKU]`).
  5. If the context does not contain statutory backing, state: *"Berdasarkan peraturan yang tersedia, dasar hukum untuk pertanyaan ini tidak ditemukan."*

### 4.4 User Interfaces (`app.py`, `cli.py`)
- **Streamlit:**
  - Multi-turn chat interface with streaming responses.
  - Side panel showing retrieved context chunks, rank score, regulation status badge, and exact Pasal text.
  - "Peraturan Aktif Saja" toggle filter.
- **CLI:**
  - Subcommands:
    - `python cli.py query "pertanyaan"`
    - `python cli.py ingest --file <path_to_pdf> --status Berlaku`
    - `python cli.py eval`

### 4.5 Evaluation Harness (`benchmark.py`)
- Contains 15 validated golden question-answer-citation triplets covering:
  - Minimum paid-up capital rules (e.g. POJK 10/2022 Pasal 8).
  - Maximum lending limits / batas maksimum pendanaan (Pasal 26).
  - Personal data handling requirements under UU PDP.
  - Negative tests (asking questions not governed by the indexed regulations).
  - Revoked rule check (verifying system does not cite superseded provisions).
- Calculates metrics:
  - Citation Recall (% of retrieved contexts containing the ground truth Pasal).
  - Faithfulness (% of answer claims backed by context).
  - Correct refusal on out-of-scope questions.

---

## 5. Sharing & Friend Setup
To ensure any colleague or recruiter can clone and run in under 3 minutes:
1. Include `data/sample/` with already-parsed JSON chunks of:
   - POJK 10/POJK.05/2022 (P2P Lending)
   - SEOJK 19/SEOJK.06/2023 (Penyelenggaraan LPBBTI)
   - UU 27/2022 (Pelindungan Data Pribadi)
2. Include pre-built CLI command `python cli.py bootstrap` to index sample data without requiring manual PDF extraction.
3. Only requires `.env` with `NINEROUTER_API_KEY`.

---

## 6. Verification & Quality Gates
- **Unit test suite:** Test legal parser regex across complex multi-clause articles.
- **Retrieval test:** Verify query "Pasal 8 POJK 10/2022" ranks the exact article in top 1.
- **CLI test:** Verify `cli.py query` returns expected output and exit code 0.
- **UI check:** Verify Streamlit launches without syntax or import errors.
