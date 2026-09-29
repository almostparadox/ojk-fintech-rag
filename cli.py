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
    console.print(f"Parsed [cyan]{len(chunks)}[/cyan] pasal from [bold]{file_path.name}[/bold]")
    indexer = LegalIndexer()
    indexer.index_chunks(chunks)
    console.print(f"[bold green]✓ Berhasil mengindeks {file_path.name}![/bold green]")

def cmd_eval(args):
    console.print("[bold cyan]Menjalankan Benchmark Evaluasi Kepatuhan Hukum...[/bold cyan]")
    report = run_evaluation_benchmark()
    console.print(Panel(report, title="Hasil Benchmark Evaluasi"))

def build_parser() -> argparse.ArgumentParser:
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

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
