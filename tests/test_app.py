import py_compile
from pathlib import Path

def test_streamlit_app_file_valid():
    app_file = Path(__file__).parent.parent / "app.py"
    assert app_file.exists()
    content = app_file.read_text(encoding="utf-8")
    assert "import streamlit as st" in content
    assert "HybridSearcher" in content
    assert "LegalGenerator" in content
    assert "LegalIndexer" in content

def test_streamlit_app_syntax():
    app_file = Path(__file__).parent.parent / "app.py"
    # Ensure app.py compiles cleanly with no syntax errors
    py_compile.compile(str(app_file), doraise=True)
