# tests/test_generation.py
from unittest.mock import MagicMock, patch
from src.config import LegalChunk
from src.generation.prompt import build_system_prompt, build_user_prompt
from src.generation.client import LegalGenerator

def test_prompt_formatting():
    chunks = [
        LegalChunk(
            id="POJK_10_2022_PASAL_8",
            reg_id="POJK 10/POJK.05/2022",
            reg_title="LPBBTI",
            status="Berlaku",
            bab="BAB III",
            pasal="Pasal 8",
            legal_ref="POJK 10/POJK.05/2022 Pasal 8",
            content="Modal disetor minimal Rp25 miliar."
        )
    ]
    sys_prompt = build_system_prompt()
    user_prompt = build_user_prompt(query="Berapa modal fintech?", contexts=chunks)
    
    assert "POJK 10/POJK.05/2022 Pasal 8" in user_prompt
    assert "Rp25 miliar" in user_prompt
    assert "Berlaku" in user_prompt
    assert "dasar hukum" in sys_prompt.lower()

def test_legal_generator_init():
    gen = LegalGenerator(api_key="custom_key", base_url="https://custom.router/v1", model="test-model")
    assert gen.api_key == "custom_key"
    assert gen.base_url == "https://custom.router/v1"
    assert gen.model == "test-model"

def test_generate_response_mocked():
    gen = LegalGenerator(api_key="mock_key")
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Berdasarkan POJK 10/2022 Pasal 8, modal disetor Rp25 miliar."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    gen._client = mock_client
    chunks = [
        LegalChunk(
            id="POJK_10_2022_PASAL_8",
            reg_id="POJK 10/POJK.05/2022",
            reg_title="LPBBTI",
            status="Berlaku",
            bab="BAB III",
            pasal="Pasal 8",
            legal_ref="POJK 10/POJK.05/2022 Pasal 8",
            content="Modal disetor minimal Rp25 miliar."
        )
    ]
    ans = gen.generate_response("Berapa modal fintech?", chunks)
    assert "Rp25 miliar" in ans
    assert mock_client.chat.completions.create.called
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert len(kwargs["messages"]) == 2

def test_stream_response_mocked():
    gen = LegalGenerator(api_key="mock_key")
    mock_client = MagicMock()
    
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Modal "))]
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock(delta=MagicMock(content="Rp25 miliar."))]
    mock_chunk3 = MagicMock()
    mock_chunk3.choices = [MagicMock(delta=MagicMock(content=None))]
    
    mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
    gen._client = mock_client

    chunks = []
    tokens = list(gen.stream_response("test query", chunks))
    assert "".join(tokens) == "Modal Rp25 miliar."

def test_stream_response_empty_choices():
    gen = LegalGenerator(api_key="mock_key")
    mock_client = MagicMock()
    mock_chunk_empty = MagicMock()
    mock_chunk_empty.choices = []
    mock_chunk_valid = MagicMock()
    mock_chunk_valid.choices = [MagicMock(delta=MagicMock(content="OK"))]
    mock_client.chat.completions.create.return_value = [mock_chunk_empty, mock_chunk_valid]
    gen._client = mock_client
    tokens = list(gen.stream_response("test", []))
    assert tokens == ["OK"]
