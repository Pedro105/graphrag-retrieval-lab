"""Application configuration, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = "INFO"

    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: SecretStr | None = None

    # Optional: fully cached runs need no Anthropic key, and the default
    # local embeddings need no OpenAI key. Missing keys fail at first use.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    embedding_provider: Literal["local", "openai"] = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimension: int = 1536
    # One setting per LLM use so each can be pointed at a different model.
    extraction_model: str = "claude-haiku-4-5-20251001"
    classifier_model: str = "claude-haiku-4-5-20251001"
    synthesis_model: str = "claude-haiku-4-5-20251001"
    # Temperature 0 so extraction and judging are as repeatable as the API allows.
    llm_temperature: float = 0.0
    # Disk cache for every LLM call (see llm/client.py).
    llm_cache_enabled: bool = True
    llm_cache_dir: Path = Path(".cache/llm")

    # Entity-resolution cosine thresholds. Hand-set; spot checks show they
    # separate name variants from distinct entities for both bge-small and
    # text-embedding-3-small, but they are not yet calibrated on data.
    vector_merge_threshold: float = 0.92
    vector_candidate_threshold: float = 0.80


@lru_cache
def get_settings() -> Settings:
    return Settings()
