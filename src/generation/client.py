# src/generation/client.py
from typing import Generator, List, Optional
from openai import OpenAI
from src.config import LegalChunk, settings
from src.generation.prompt import build_system_prompt, build_user_prompt

class LegalGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or settings.NINEROUTER_API_KEY
        self.base_url = base_url or settings.NINEROUTER_BASE_URL
        self.model = model or settings.DEFAULT_MODEL
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key or "dummy_key",
                base_url=self.base_url
            )
        return self._client

    def generate_response(self, query: str, contexts: List[LegalChunk]) -> str:
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(query, contexts)}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content or ""

    def stream_response(self, query: str, contexts: List[LegalChunk]) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(query, contexts)}
        ]
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
