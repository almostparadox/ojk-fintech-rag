import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()


class Settings(BaseModel):
    LLM_API_KEY: str = Field(default="")
    LLM_BASE_URL: str = Field(default="https://api.openai.com/v1")
    DEFAULT_MODEL: str = Field(default="gpt-4o-mini")
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    STORAGE_DIR: Path = Field(default_factory=lambda: Path(os.getenv("STORAGE_DIR", "./storage")).resolve())
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")

    # Alias fields
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_credentials(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            data = {}

        api_key = (
            data.get("LLM_API_KEY")
            or data.get("OPENAI_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )

        base_url = (
            data.get("LLM_BASE_URL")
            or data.get("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )

        default_model = (
            data.get("DEFAULT_MODEL")
            or os.getenv("DEFAULT_MODEL")
            or "gpt-4o-mini"
        )

        data["LLM_API_KEY"] = api_key
        data["LLM_BASE_URL"] = base_url
        data["DEFAULT_MODEL"] = default_model
        data["OPENAI_API_KEY"] = api_key
        data["OPENAI_BASE_URL"] = base_url
        return data


class LegalChunk(BaseModel):
    id: str
    reg_id: str
    reg_title: str
    status: str = "Berlaku"  # "Berlaku", "Diubah", "Dicabut"
    bab: str = ""
    pasal: str
    legal_ref: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_search_text(self) -> str:
        status_tag = f"[{self.status.upper()}]"
        return f"[{self.reg_id}] {status_tag} {self.bab} - {self.pasal}\n{self.content}"


settings = Settings()
