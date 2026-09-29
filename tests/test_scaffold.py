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
