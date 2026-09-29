import subprocess
import sys
from pathlib import Path
import pytest

def test_cli_help():
    result = subprocess.run([sys.executable, "cli.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "bootstrap" in result.stdout
    assert "query" in result.stdout
    assert "ingest" in result.stdout
    assert "eval" in result.stdout

def test_cli_subcommands_help():
    for subcmd in ["bootstrap", "query", "ingest", "eval"]:
        result = subprocess.run([sys.executable, "cli.py", subcmd, "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "help" in result.stdout or subcmd in result.stdout

def test_cli_bootstrap_and_query_no_llm(tmp_path, monkeypatch):
    import cli
    from src.config import settings

    # Point storage to tmp_path
    storage_dir = tmp_path / "storage"
    monkeypatch.setattr(settings, "STORAGE_DIR", storage_dir)

    # 1. Test bootstrap
    class ArgsBootstrap:
        pass
    cli.cmd_bootstrap(ArgsBootstrap())

    assert (storage_dir / "bm25.pkl").exists()
    assert (storage_dir / "lancedb").exists()

    # 2. Test query with --no-llm
    class ArgsQuery:
        query = "modal disetor fintech"
        top_k = 2
        include_revoked = False
        no_llm = True

    cli.cmd_query(ArgsQuery())

def test_cli_ingest(tmp_path, monkeypatch):
    import cli
    from src.config import settings

    storage_dir = tmp_path / "storage_ingest"
    monkeypatch.setattr(settings, "STORAGE_DIR", storage_dir)

    test_file = tmp_path / "sample_rule.txt"
    test_file.write_text("Pasal 1\nKetentuan umum fintech.", encoding="utf-8")

    class ArgsIngest:
        file = str(test_file)
        reg_id = "POJK 99/2026"
        reg_title = "Fintech Sandbox"
        status = "Berlaku"

    cli.cmd_ingest(ArgsIngest())
    assert (storage_dir / "bm25.pkl").exists()

def test_cli_ingest_missing_file(tmp_path, monkeypatch, capsys):
    import cli
    from src.config import settings

    class ArgsIngestMissing:
        file = str(tmp_path / "does_not_exist.pdf")
        reg_id = "POJK 99/2026"
        reg_title = "Fintech Sandbox"
        status = "Berlaku"

    cli.cmd_ingest(ArgsIngestMissing())

def test_cli_eval(tmp_path, monkeypatch):
    import cli
    from src.config import settings

    storage_dir = tmp_path / "storage_eval"
    monkeypatch.setattr(settings, "STORAGE_DIR", storage_dir)

    class ArgsEval:
        pass

    cli.cmd_eval(ArgsEval())
