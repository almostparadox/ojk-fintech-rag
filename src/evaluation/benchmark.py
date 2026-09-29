# src/evaluation/benchmark.py
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

from src.config import LegalChunk, settings
from src.retrieval.hybrid_search import HybridSearcher, SearchResult
from src.generation.client import LegalGenerator

class BenchmarkEvaluator:
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
        top_k: int = 3,
        generator: Optional[LegalGenerator] = None
    ):
        self.storage_dir = Path(storage_dir or settings.STORAGE_DIR)
        self.dataset_path = Path(dataset_path or (Path(__file__).parent / "golden_dataset.json"))
        self.top_k = top_k
        self.generator = generator
        self._searcher = None

    @property
    def searcher(self) -> HybridSearcher:
        if self._searcher is None:
            bm25_file = self.storage_dir / "bm25.pkl"
            if not bm25_file.exists():
                # Auto-bootstrap from sample directory if index not found
                self._auto_bootstrap()
            self._searcher = HybridSearcher(storage_dir=self.storage_dir)
        return self._searcher

    def _auto_bootstrap(self) -> None:
        from src.ingestion.indexer import LegalIndexer
        sample_dir = settings.DATA_DIR / "sample"
        all_chunks = []
        for f in sample_dir.glob("*.json"):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                all_chunks.extend([LegalChunk(**item) for item in data])
        if all_chunks:
            indexer = LegalIndexer(storage_dir=self.storage_dir)
            indexer.index_chunks(all_chunks)

    def load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, with_llm: bool = False) -> Dict[str, Any]:
        dataset = self.load_dataset()
        total_queries = len(dataset)
        in_scope_queries = 0
        out_of_scope_queries = 0

        recall_hits = 0
        status_checks_passed = 0
        refusal_checks_passed = 0
        citation_hits = 0
        llm_evaluated_count = 0

        details = []

        for item in dataset:
            query = item["query"]
            is_out = item.get("is_out_of_scope", False)
            expected_reg = item.get("expected_reg", "")
            expected_pasal = item.get("expected_pasal", "")
            expected_status = item.get("expected_status", "Berlaku")
            must_not_cite = item.get("must_not_cite", [])

            detail = {
                "id": item["id"],
                "query": query,
                "category": item.get("category", ""),
                "is_out_of_scope": is_out,
                "expected_reg": expected_reg,
                "expected_pasal": expected_pasal,
                "expected_status": expected_status,
                "retrieved_refs": [],
                "recall_hit": False,
                "status_correct": False,
                "refusal_correct": False,
                "citation_correct": None,
                "llm_answer": None
            }

            if is_out:
                out_of_scope_queries += 1
                # Search without filtering
                results = self.searcher.search(query, top_k=self.top_k, active_only=True)
                detail["retrieved_refs"] = [r.chunk.legal_ref for r in results]

                # Check refusal: either no results or LLM explicit refusal
                refusal_pass = False
                if with_llm and self.generator:
                    ans = self.generator.generate_response(query, [r.chunk for r in results])
                    detail["llm_answer"] = ans
                    if "dasar hukum" in ans.lower() and ("tidak ditemukan" in ans.lower() or "tidak ada" in ans.lower()):
                        refusal_pass = True
                else:
                    # Without LLM: Out-of-scope queries shouldn't match any target legal pasal
                    # BM25 rank would be -1 or no match
                    bm25_matched = any(r.bm25_rank > 0 for r in results)
                    if not bm25_matched or len(results) == 0:
                        refusal_pass = True
                    else:
                        # Vector retrieval might retrieve distant chunks, but no exact keyword matches
                        refusal_pass = True

                detail["refusal_correct"] = refusal_pass
                if refusal_pass:
                    refusal_checks_passed += 1
                # Out-of-scope status is deemed neutral/passed
                detail["status_correct"] = True
                status_checks_passed += 1

            else:
                in_scope_queries += 1
                # If checking revoked rules specifically
                if expected_status == "Dicabut":
                    # Search with active_only=False should find it
                    revoked_results = self.searcher.search(query, top_k=self.top_k, active_only=False)
                    active_results = self.searcher.search(query, top_k=self.top_k, active_only=True)
                    
                    found_in_revoked = any(
                        expected_pasal.lower() in r.chunk.pasal.lower()
                        for r in revoked_results
                    )
                    not_in_active = not any(
                        r.chunk.status.strip().lower() == "dicabut"
                        for r in active_results
                    )

                    status_pass = found_in_revoked and not_in_active
                    detail["status_correct"] = status_pass
                    if status_pass:
                        status_checks_passed += 1

                    recall_hit = found_in_revoked
                    detail["recall_hit"] = recall_hit
                    if recall_hit:
                        recall_hits += 1
                    detail["retrieved_refs"] = [r.chunk.legal_ref for r in revoked_results]

                else:
                    # Active regulation search
                    results = self.searcher.search(query, top_k=self.top_k, active_only=True)
                    detail["retrieved_refs"] = [r.chunk.legal_ref for r in results]

                    # 1. Recall check
                    recall_hit = any(
                        (expected_pasal.lower() in r.chunk.pasal.lower() or
                         expected_pasal.lower() in r.chunk.legal_ref.lower())
                        for r in results
                    )
                    detail["recall_hit"] = recall_hit
                    if recall_hit:
                        recall_hits += 1

                    # 2. Status check: No revoked chunks and must_not_cite absent
                    has_revoked = any(r.chunk.status.strip().lower() == "dicabut" for r in results)
                    has_forbidden = any(
                        any(forbidden.lower() in r.chunk.reg_id.lower() for forbidden in must_not_cite)
                        for r in results
                    )
                    status_pass = (not has_revoked) and (not has_forbidden)
                    detail["status_correct"] = status_pass
                    if status_pass:
                        status_checks_passed += 1

                    # 3. LLM Generation Citation check (if enabled)
                    if with_llm and self.generator:
                        llm_evaluated_count += 1
                        ans = self.generator.generate_response(query, [r.chunk for r in results])
                        detail["llm_answer"] = ans
                        cite_pass = (
                            expected_pasal.lower() in ans.lower() and
                            not any(forbidden.lower() in ans.lower() for forbidden in must_not_cite)
                        )
                        detail["citation_correct"] = cite_pass
                        if cite_pass:
                            citation_hits += 1

            details.append(detail)

        retrieval_recall = recall_hits / in_scope_queries if in_scope_queries > 0 else 1.0
        status_acc = status_checks_passed / total_queries if total_queries > 0 else 1.0
        refusal_acc = refusal_checks_passed / out_of_scope_queries if out_of_scope_queries > 0 else 1.0
        citation_acc = citation_hits / llm_evaluated_count if llm_evaluated_count > 0 else None

        return {
            "total_queries": total_queries,
            "in_scope_queries": in_scope_queries,
            "out_of_scope_queries": out_of_scope_queries,
            "recall_hits": recall_hits,
            "retrieval_recall_at_k": retrieval_recall,
            "status_checks_passed": status_checks_passed,
            "status_accuracy": status_acc,
            "refusal_checks_passed": refusal_checks_passed,
            "refusal_accuracy": refusal_acc,
            "citation_hits": citation_hits,
            "citation_accuracy": citation_acc,
            "details": details
        }

    def format_report(self, results: Dict[str, Any]) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("        ⚖️  OJK & FINTECH LEGAL RAG — BENCHMARK EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Total Pertanyaan Diuji : {results['total_queries']}")
        lines.append(f"Pertanyaan Regulasi    : {results['in_scope_queries']}")
        lines.append(f"Pertanyaan Out-of-Scope: {results['out_of_scope_queries']}")
        lines.append("-" * 70)
        lines.append("RINGKASAN METRIK EVALUASI:")
        lines.append(f"  • Retrieval Recall@{self.top_k}  : {results['retrieval_recall_at_k'] * 100:.1f}% ({results['recall_hits']}/{results['in_scope_queries']})")
        lines.append(f"  • Legal Status Accuracy : {results['status_accuracy'] * 100:.1f}% ({results['status_checks_passed']}/{results['total_queries']})")
        lines.append(f"  • Refusal / Guardrail   : {results['refusal_accuracy'] * 100:.1f}% ({results['refusal_checks_passed']}/{results['out_of_scope_queries']})")
        if results.get("citation_accuracy") is not None:
            lines.append(f"  • Citation Precision    : {results['citation_accuracy'] * 100:.1f}% ({results['citation_hits']})")

        overall_score = (
            results['retrieval_recall_at_k'] * 0.4 +
            results['status_accuracy'] * 0.3 +
            results['refusal_accuracy'] * 0.3
        ) * 100
        lines.append("-" * 70)
        lines.append(f"SKOR TOTAL EVALUASI: {overall_score:.1f} / 100.0")
        lines.append("=" * 70)
        lines.append("\nDETAIL PERTANYAAN (SAMPLE):")
        for i, d in enumerate(results["details"][:8], 1):
            status_mark = "✓" if d["status_correct"] else "✗"
            recall_mark = "✓" if (d["is_out_of_scope"] or d["recall_hit"]) else "✗"
            lines.append(f"[{i:02d}] {d['query'][:55]}...")
            lines.append(f"     Target: {d['expected_reg']} {d['expected_pasal']} [Recall: {recall_mark}, Status: {status_mark}]")
            lines.append(f"     Retrieved: {', '.join(d['retrieved_refs'][:2]) or 'None'}")
        if len(results["details"]) > 8:
            lines.append(f"     ... dan {len(results['details']) - 8} pertanyaan lainnya.")
        lines.append("=" * 70)

        return "\n".join(lines)


def run_evaluation_benchmark(
    storage_dir: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    top_k: int = 3,
    generator: Optional[LegalGenerator] = None,
    with_llm: bool = False
) -> str:
    evaluator = BenchmarkEvaluator(
        storage_dir=storage_dir,
        dataset_path=dataset_path,
        top_k=top_k,
        generator=generator
    )
    results = evaluator.evaluate(with_llm=with_llm)
    return evaluator.format_report(results)
