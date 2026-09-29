from pathlib import Path
import pytest
import pypdf
from src.ingestion.pdf_loader import load_document

def test_load_text_document(tmp_path):
    doc_path = tmp_path / "sample_rule.txt"
    doc_path.write_text("Pasal 1\nKetentuan umum mengenai fintech.", encoding="utf-8")
    
    text = load_document(doc_path)
    assert "Pasal 1" in text
    assert "fintech" in text

def test_load_missing_file_raises(tmp_path):
    missing_path = tmp_path / "non_existent.pdf"
    with pytest.raises(FileNotFoundError):
        load_document(missing_path)

def test_load_pdf_document(tmp_path):
    # Create a real minimal PDF
    pdf_path = tmp_path / "sample_rule.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    
    # Should load without error and return string (even if empty for blank page)
    result = load_document(pdf_path)
    assert isinstance(result, str)
