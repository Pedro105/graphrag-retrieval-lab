"""Application configuration, loaded from environment variables / .env."""

from functools import lru_cache

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

    anthropic_api_key: SecretStr
    openai_api_key: SecretStr

    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    # One setting per LLM use so each can be pointed at a different model.
    extraction_model: str = "claude-haiku-4-5-20251001"
    classifier_model: str = "claude-haiku-4-5-20251001"
    synthesis_model: str = "claude-haiku-4-5-20251001"
    vector_merge_threshold: float = 0.92
    vector_candidate_threshold: float = 0.80


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
