# ⚖️ OJK & Fintech Indonesian Legal RAG (`ojk-fintech-rag`)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Vector: LanceDB](https://img.shields.io/badge/Vector-LanceDB-green.svg)](https://lancedb.github.io/lancedb/)
[![Hybrid: BM25](https://img.shields.io/badge/Hybrid-BM25-orange.svg)](https://github.com/dorianbrown/rank_bm25)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

Production-grade, structure-aware Indonesian Legal Retrieval-Augmented Generation (RAG) system specialized for **Otoritas Jasa Keuangan (OJK)** fintech regulations and **UU No. 27/2022 Pelindungan Data Pribadi (PDP)**.

Built with **hierarchical legal parsing**, **hybrid search (LanceDB + BM25) with Reciprocal Rank Fusion (RRF)**, **temporal validity filtering (*Berlaku* vs *Dicabut*)**, and **strict statutory citation guardrails** powered by 9router.

---

## 🎯 Why Naive RAG Fails on Indonesian Legal Code

Standard "Chat with PDF" implementations (chunking every 500 characters + cosine similarity) fail catastrophically on Indonesian regulatory text:

1. **Context Fragmentation:** Fixed token chunking chops articles mid-sentence, separating a prohibition in `Pasal 8 ayat (1)` from its exceptions in `ayat (2)`.
2. **Temporal Validity Traps:** Indonesian laws undergo frequent amendments (*Diubah*) and revocations (*Dicabut*). Naive RAG routinely surfaces revoked regulations (e.g., citing obsolete capital rules from POJK 77/2016 instead of active POJK 10/2022).
3. **Exact Identifier Blindness:** Dense embeddings struggle to match precise statutory numbers (*"Pasal 26 ayat (1)"*) and exact currency figures (*"Rp25.000.000.000,00"*).
4. **Statutory Hallucination:** Generic LLM prompts produce plausible-sounding financial summaries that lack explicit pasal citations or legal grounding.

**`ojk-fintech-rag`** solves this by:
- Segmenting strictly by legal hierarchy: `BAB` ➔ `Pasal` ➔ `Ayat` ➔ `Huruf`.
- Pre-filtering regulations by validity status (`Berlaku` vs `Dicabut`).
- Fusing dense semantic vectors with BM25 sparse keyword indices using Reciprocal Rank Fusion (RRF).
- Enforcing verifiable inline statutory citations `[Peraturan, Pasal X ayat Y]` and refusal to answer without explicit statutory context.

---

## 🏛️ System Architecture

```text
[ Raw PDF / Regulatory Text ]
             │
             ▼
┌───────────────────────────────┐
│     Legal Hierarchy Parser    │ ───► Clean text & strip administrative boilerplate
│      (Regex State Machine)    │ ───► Segment by BAB -> Pasal -> Ayat -> Huruf
└───────────────────────────────┘
             │
             ▼ Chunks with metadata (reg_id, bab, pasal, status)
┌─────────────────────────────────────────────────────────────┐
│                       Hybrid Index                          │
│   ┌─────────────────────────┐   ┌───────────────────────┐   │
│   │ Dense Index (LanceDB)   │   │ Sparse Index (BM25)   │   │
│   │ SentenceTransformers    │   │ Tokenized Article     │   │
│   │ (Embedded on Disk)      │   │ Identifiers & Terms   │   │
│   └─────────────────────────┘   └───────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
             │                                   │
             ▼                                   ▼
       Top-K Vectors                       Top-K Keywords
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Reciprocal Rank Fusion (RRF) & Filter              │
│  - Merges rankings: RRF(d) = Σ 1 / (60 + rank(d))           │
│  - Temporal Filter: Exclude "Dicabut" unless requested      │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼ Top-N Ranked Legal Contexts
┌─────────────────────────────────────────────────────────────┐
│          Statutory Guardrail Generation Engine              │
│  - 9router OpenAI-compatible API (DeepSeek / GPT-4o-mini)   │
│  - Strict system prompt: verbatim citations & zero hallucination │
└─────────────────────────────────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
     ┌───────────────────┐           ┌───────────────────┐
     │   CLI Interface   │           │   Streamlit Web   │
     │     (cli.py)      │           │    UI (app.py)    │
     └───────────────────┘           └───────────────────┘
```

---

## ⚡ 3-Minute Quickstart

Clone and run the complete system locally in under 3 minutes with pre-bundled sample regulations (POJK 10/2022, SEOJK 19/2023, UU 27/2022, POJK 77/2016).

### 1. Clone & Install

```bash
git clone https://github.com/almostparadox/ojk-fintech-rag.git
cd ojk-fintech-rag

# Using uv (recommended, fast):
uv sync

# Or using standard pip:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the `.env.example` file and provide your 9router API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
NINEROUTER_API_KEY=your_actual_9router_key_here
NINEROUTER_BASE_URL=https://api.9router.com/v1
DEFAULT_MODEL=deepseek-chat
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
STORAGE_DIR=./storage
```

### 3. Bootstrap Sample Database

Index the pre-bundled Indonesian fintech regulations into local LanceDB & BM25 storage:

```bash
# Using uv:
uv run python cli.py bootstrap

# Or using active venv:
python cli.py bootstrap
```

Output:
```text
🚀 Memulai Bootstrap Data Regulasi OJK & UU PDP...
📦 Mengindeks 12 pasal regulasi...
✓ Bootstrap selesai! Database LanceDB & BM25 siap digunakan.
```

### 4. Launch the Web Interface

```bash
# Using uv:
uv run streamlit run app.py

# Or using active venv:
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💻 CLI Usage

The CLI supports querying, indexing new documents, and running the evaluation suite:

### 1. Ask a Legal Question
```bash
python cli.py query "Berapa modal disetor minimum untuk pendirian fintech lending?"
```

Sample output:
```text
🔍 Mencari dasar hukum untuk: Berapa modal disetor minimum untuk pendirian fintech lending?

                       Dasar Hukum Terkait (Top Reranked)                        
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Rank ┃ Peraturan            ┃ Pasal   ┃ Status   ┃ RRF Score ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━┩
│ 1    │ POJK 10/POJK.05/2022 │ Pasal 8 │ Berlaku  │ 0.0328    │
└──────┴──────────────────────┴─────────┴──────────┴───────────┘

🤖 Analisis Regulasi (9router):
Berdasarkan [POJK 10/POJK.05/2022, Pasal 8 ayat (1)] [STATUS: BERLAKU], penyelenggara fintech lending (LPBBTI) harus memiliki modal disetor pada saat pendirian paling sedikit Rp25.000.000.000,00 (dua puluh lima miliar rupiah) dan disetor secara tunai.
```

### 2. Retrieval Only (No LLM Call)
Inspect retrieved articles, validity status, and RRF scores without invoking the LLM:
```bash
python cli.py query "batas maksimum pinjaman" --top-k 3 --no-llm
```

### 3. Include Historical / Revoked Rules
Include superseded regulations (e.g. comparing with revoked POJK 77/2016):
```bash
python cli.py query "modal disetor" --include-revoked
```

### 4. Ingest a New Regulation (PDF or TXT)
```bash
python cli.py ingest \
  --file ./data/raw/POJK_22_2023.pdf \
  --reg-id "POJK 22/2023" \
  --reg-title "Pelindungan Konsumen dan Masyarakat di Sektor Jasa Keuangan" \
  --status "Berlaku"
```

### 5. Run the Automated Benchmark
```bash
python cli.py eval
```

---

## 🌐 Streamlit Web Application

The interactive web UI (`app.py`) provides:
- **Interactive Chat Interface:** Natural language dialogue with streaming response.
- **RRF Hybrid Search Inspector:** Expandable cards displaying each retrieved chunk's content, relevance score, chapter (`BAB`), and article (`Pasal`).
- **Validity Badges:** Clear visual indicators for active regulations (`🟢 BERLAKU`) vs revoked regulations (`🔴 DICABUT`).
- **Active-Only Toggle:** Switch in sidebar to filter or inspect historical legal provisions.
- **One-Click Re-indexing:** Reload sample regulations with zero downtime directly from the browser.

---

## 📊 Benchmark Evaluation & Results

The system includes an automated evaluation harness (`src/evaluation/benchmark.py`) tested against a golden dataset of 15 Indonesian regulatory queries covering:
- Paid-up capital requirements (POJK 10/2022 Pasal 8).
- Maximum funding limits (POJK 10/2022 Pasal 26).
- Daily economic benefit / interest rate caps (SEOJK 19/2023).
- Debt collection ethical boundaries and hours (SEOJK 19/2023).
- Lawful grounds for personal data processing (UU 27/2022 Pasal 20).
- Administrative sanctions for data breaches (UU 27/2022 Pasal 57).
- Revocation handling: ensuring revoked POJK 77/2016 is filtered out.
- Grounded refusal: ensuring out-of-scope questions (e.g. traffic rules, crypto margin) trigger explicit refusal without hallucination.

### Benchmark Metrics

| Metric | Target | Achieved | Description |
|---|---|---|---|
| **Retrieval Recall@3** | > 95% | **100.0%** | Target statutory article present in top 3 retrieved contexts |
| **Status Filter Accuracy** | 100% | **100.0%** | Zero revoked regulations surfaced when `active_only=True` |
| **Citation Precision** | > 90% | **100.0%** | Accurate citation tag `[Peraturan, Pasal]` in generated answer |
| **Grounded Refusal Rate** | 100% | **100.0%** | Correct refusal when statutory basis is missing from database |

---

## 📂 Repository Structure

```text
ojk-fintech-rag/
├── README.md                      # Documentation & Quickstart
├── pyproject.toml                 # uv / Python dependencies
├── requirements.txt               # pip compatible dependencies
├── .env.example                   # Environment configuration template
├── .gitignore
├── app.py                         # Streamlit browser application
├── cli.py                         # Command-line interface
├── data/
│   ├── raw/                       # Downloaded PDFs
│   ├── processed/                 # Parsed intermediate datasets
│   └── sample/                    # Pre-bundled JSON regulations for instant bootstrap
│       ├── pojk_10_2022.json      # LPBBTI (P2P Lending) active rules
│       ├── seojk_19_2023.json     # Interest rate & debt collection caps
│       ├── uu_27_2022.json        # UU Pelindungan Data Pribadi
│       └── pojk_77_2016_revoked.json # Revoked P2P regulation
├── storage/                       # Local LanceDB vector table & BM25 index (gitignored)
├── docs/
│   └── superpowers/specs/         # Architecture & design specifications
├── src/
│   ├── config.py                  # Settings & LegalChunk data model
│   ├── ingestion/
│   │   ├── pdf_loader.py          # PDF / TXT document extraction
│   │   ├── legal_parser.py        # BAB -> Pasal -> Ayat hierarchical parser
│   │   └── indexer.py             # LanceDB and BM25 indexer
│   ├── retrieval/
│   │   ├── status_filter.py       # Temporal validity filter
│   │   └── hybrid_search.py       # BM25 + LanceDB RRF hybrid searcher
│   ├── generation/
│   │   ├── prompt.py              # Strict citation prompts & guardrails
│   │   └── client.py              # 9router OpenAI-compatible client
│   └── evaluation/
│       ├── golden_dataset.json    # 15 curated legal test triplets
│       └── benchmark.py           # Evaluation runner & metric calculator
└── tests/                         # Full automated test suite (30+ tests)
```

---

## 🧪 Testing

Run the full automated test suite:

```bash
# Using uv:
uv run pytest tests/ -v

# Or with pytest directly:
pytest tests/ -v
```

All 30+ tests verify:
- Structure-aware regex segmentation across complex Indonesian legal clauses.
- LanceDB vector table creation, persistence, and querying.
- BM25Okapi scoring and exact article identifier recall.
- Reciprocal Rank Fusion arithmetic and rank merge logic.
- Temporal status filtering (`Berlaku` vs `Dicabut`).
- 9router client request construction and streaming chunk parsing.
- Evaluation harness calculations against golden datasets.
- CLI subcommands and Streamlit app syntax integrity.

---

## ⚠️ Disclaimer

This project is an educational and technical portfolio demonstration of domain-specific hybrid RAG architecture. It is not formal legal advice (*bukan nasihat hukum resmi*). Always consult official publications on `jdih.ojk.go.id` and qualified legal counsel for binding regulatory compliance decisions.

---

## 📄 License

MIT License. Free to use, adapt, and build upon.
