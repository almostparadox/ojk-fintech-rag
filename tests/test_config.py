from src.config import LegalChunk, Settings


def test_settings_default_values():
    settings = Settings(NINEROUTER_API_KEY="test-key")
    assert settings.NINEROUTER_API_KEY == "test-key"
    assert settings.NINEROUTER_BASE_URL == "https://api.9router.com/v1"
    assert settings.DEFAULT_MODEL == "deepseek-chat"


def test_settings_generic_llm_values():
    settings = Settings(
        LLM_API_KEY="generic-key",
        LLM_BASE_URL="https://api.deepseek.com/v1",
        DEFAULT_MODEL="deepseek-reasoner",
    )
    assert settings.LLM_API_KEY == "generic-key"
    assert settings.LLM_BASE_URL == "https://api.deepseek.com/v1"
    assert settings.DEFAULT_MODEL == "deepseek-reasoner"
    assert settings.OPENAI_API_KEY == "generic-key"


def test_legal_chunk_model():
    chunk = LegalChunk(
        id="POJK_10_2022_PASAL_8",
        reg_id="POJK 10/POJK.05/2022",
        reg_title="Layanan Pendanaan Bersama Berbasis Teknologi Informasi",
        status="Berlaku",
        bab="BAB III PERIZINAN",
        pasal="Pasal 8",
        legal_ref="POJK 10/POJK.05/2022 Pasal 8",
        content="Pasal 8\n(1) Modal disetor minimal Rp25 miliar.",
        metadata={"tahun": 2022},
    )
    assert chunk.status == "Berlaku"
    assert "Pasal 8" in chunk.content
    assert chunk.to_search_text().startswith("[POJK 10/POJK.05/2022]")
