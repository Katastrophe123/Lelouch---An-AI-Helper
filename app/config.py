"""Application settings, loaded from the environment (12-factor style)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["stub", "openai", "huggingface"]
Environment = Literal["local", "test", "production"]


class Settings(BaseSettings):
    """Runtime configuration. See ``.env.example`` for the full list."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "ai-homework-helper"
    environment: Environment = "local"
    log_level: str = "INFO"
    log_json: bool = False

    # --- Database ---------------------------------------------------------
    database_url: str = "postgresql+asyncpg://tutor:tutor@localhost:5432/tutor"
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # --- Auth -------------------------------------------------------------
    jwt_secret: str = "insecure-local-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720
    auth_password: str = "homework"

    # --- LLM --------------------------------------------------------------
    llm_provider: LLMProvider = "stub"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm_timeout_seconds: float = 60.0
    llm_max_attempts: int = Field(default=3, ge=1, le=10)
    llm_backoff_base_seconds: float = Field(default=0.5, ge=0.0)

    # --- Tutor behaviour --------------------------------------------------
    self_check_min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    max_solve_retries: int = Field(default=2, ge=0, le=5)
    max_graph_steps: int = Field(default=14, ge=4, le=100)
    history_window: int = Field(default=8, ge=0, le=50)

    # --- Pagination -------------------------------------------------------
    default_page_size: int = Field(default=20, ge=1, le=200)
    max_page_size: int = Field(default=100, ge=1, le=500)

    # --- Frontend ---------------------------------------------------------
    api_base_url: str = "http://localhost:8000"

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so the env is parsed once per process."""
    return Settings()
