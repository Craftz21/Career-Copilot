"""
Single source of truth for all application configuration.
Every env var is declared here exactly once.
No other file calls os.getenv() directly.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_secret_key: str = "change-me-in-production"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "info"

    # ── Database ───────────────────────────────────────────────────────────────
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/career_copilot"

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── AI: LLM ───────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"

    # ── AI: Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Skill Extraction ──────────────────────────────────────────────────────
    skill_extraction_threshold: float = 0.38
    skill_extraction_top_k: int = 30

    # ── File Upload ───────────────────────────────────────────────────────────
    max_upload_size_mb: int = 5

    # ── Sessions ──────────────────────────────────────────────────────────────
    session_ttl_days: int = 7

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_upload_per_hour: int = 10
    rate_limit_roadmap_per_hour: int = 5

    # ── Monitoring ────────────────────────────────────────────────────────────
    sentry_dsn: str = ""

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def debug(self) -> bool:
        return self.app_env == "development"

    @property
    def environment(self) -> str:
        return self.app_env

    @property
    def celery_broker_url(self) -> str:
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        return self.redis_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance. Call this everywhere instead of Settings()."""
    return Settings()
