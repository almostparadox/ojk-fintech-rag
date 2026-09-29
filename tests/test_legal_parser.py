from src.ingestion.legal_parser import LegalParser

SAMPLE_LEGAL_TEXT = """
PERATURAN OTORITAS JASA KEUANGAN
NOMOR 10 /POJK.05/2022
TENTANG
LAYANAN PENDANAAN BERSAMA BERBASIS TEKNOLOGI INFORMASI

- 1 -
Salinan sesuai dengan aslinya

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
    assert "- 1 -" not in p8.content
    assert "Salinan sesuai dengan aslinya" not in p8.content

def test_fallback_when_no_pasal():
    parser = LegalParser()
    chunks = parser.parse_text(
        text="Surat Edaran OJK tanpa pasal formal.",
        reg_id="SEOJK 19/2023",
        reg_title="SEOJK Penyelenggaraan",
        status="Berlaku"
    )
    assert len(chunks) == 1
    assert chunks[0].pasal == "Semua"
    assert chunks[0].legal_ref == "SEOJK 19/2023 Dokumen Lengkap"
    assert "Surat Edaran OJK" in chunks[0].content

def test_no_next_bab_in_pasal_content():
    parser = LegalParser()
    chunks = parser.parse_text(
        text=SAMPLE_LEGAL_TEXT,
        reg_id="POJK 10/POJK.05/2022",
        reg_title="Layanan Pendanaan Bersama Berbasis Teknologi Informasi",
    )
    p1 = next(c for c in chunks if c.pasal == "Pasal 1")
    assert "BAB III" not in p1.content
    assert "PERIZINAN DAN KELEMBAGAAN" not in p1.content
