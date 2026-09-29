# src/config.py
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    NINEROUTER_API_KEY: str = Field(default_factory=lambda: os.getenv("NINEROUTER_API_KEY", ""))
    NINEROUTER_BASE_URL: str = Field(default_factory=lambda: os.getenv("NINEROUTER_BASE_URL", "https://api.9router.com/v1"))
    DEFAULT_MODEL: str = Field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "deepseek-chat"))
    EMBEDDING_MODEL: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    STORAGE_DIR: Path = Field(default_factory=lambda: Path(os.getenv("STORAGE_DIR", "./storage")).resolve())
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")

class LegalChunk(BaseModel):
    id: str
    reg_id: str
    reg_title: str
    status: str = "Berlaku"  # "Berlaku", "Diubah", "Dicabut"
    bab: str = ""
    pasal: str
    legal_ref: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_search_text(self) -> str:
        status_tag = f"[{self.status.upper()}]"
        return f"[{self.reg_id}] {status_tag} {self.bab} - {self.pasal}\n{self.content}"

settings = Settings()
