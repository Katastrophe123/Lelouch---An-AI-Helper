"""Application settings, loaded from the environment (12-factor style)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["stub", "openai", "huggingface"]
Environment = Literal["local", "test", "production"]
AgentBackend = Literal["claude_sdk", "fake"]
AgentEffort = Literal["low", "medium", "high", "xhigh", "max"]
AgentThinking = Literal["adaptive", "disabled"]
EmbeddingBackend = Literal["fastembed", "hashing"]


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

    # --- Agent mode (Claude Agent SDK) -----------------------------------
    # "claude_sdk" runs the real Claude engine. It authenticates with your own
    # Claude login (`claude` CLI) or, if ANTHROPIC_API_KEY is set, the API key.
    # "fake" is an offline scripted agent for tests and demos.
    agent_backend: AgentBackend = "claude_sdk"
    agent_name: str = "Lelouch"
    agent_model: str = "claude-opus-5-5"
    agent_models: list[str] = Field(
        default_factory=lambda: ["claude-opus-5-5", "claude-fable-5-2", "claude-sonnet-5-5"]
    )
    agent_effort: AgentEffort = "high"
    agent_thinking: AgentThinking = "adaptive"
    agent_max_turns: int = Field(default=40, ge=1, le=500)
    # Hard spend cap per turn, in USD at API list prices. Protects the monthly
    # Agent SDK credit from one runaway research loop.
    agent_max_budget_usd: float = Field(default=3.0, gt=0.0)
    agent_timeout_seconds: float = Field(default=900.0, gt=0.0)
    agent_workspace_dir: str = "./workspace"
    agent_enable_web: bool = True
    agent_enable_playwright: bool = True
    agent_playwright_command: str = "npx -y @playwright/mcp@latest --headless --isolated"
    # Extra MCP servers, in the same JSON shape as Claude Code's `.mcp.json`.
    agent_mcp_config_path: str | None = "mcp_servers.json"
    agent_cli_path: str | None = None

    # --- Knowledge base (RAG over your own files) ---------------------------
    embedding_backend: EmbeddingBackend = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rag_chunk_chars: int = Field(default=1200, ge=200, le=8000)
    rag_chunk_overlap: int = Field(default=200, ge=0, le=2000)
    rag_top_k: int = Field(default=6, ge=1, le=50)
    max_upload_mb: float = Field(default=20.0, gt=0.0)

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

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so the env is parsed once per process."""
    return Settings()
