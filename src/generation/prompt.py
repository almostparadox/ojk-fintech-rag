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
