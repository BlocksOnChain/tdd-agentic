"""Application configuration loaded from environment via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    openai_api_key: str = ""
    # When set, all ``openai/<model>`` slugs use this host (llama.cpp, LM Studio,
    # vLLM, etc.). Example: http://127.0.0.1:8080/v1
    # From Docker Desktop (backend in a container): use host.docker.internal, not 127.0.0.1.
    openai_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("openai_base_url", "openai_api_base"),
    )
    # Seconds; raise for slow local inference.
    openai_request_timeout: float = 120.0
    anthropic_api_key: str = ""
    pm_model: str = "anthropic/claude-sonnet-4-6"
    researcher_model: str = "openai/gpt-4o"
    lead_model: str = "anthropic/claude-sonnet-4-6"
    dev_model: str = "anthropic/claude-sonnet-4-6"
    # Optional overrides for product devs (default to dev_model when unset in env).
    backend_dev_model: str | None = None
    frontend_dev_model: str | None = None
    grader_model: str = "anthropic/claude-haiku-4-5"

    # Web search backing for the ``web_search`` tool (researcher).
    # anthropic — Claude ``bind_tools`` with server-side search (needs credits).
    # tavily — Tavily REST API only (pairs with local / non-Anthropic setups).
    # auto — use tavily when TAVILY_API_KEY is set, else anthropic when ANTHROPIC_API_KEY is set.
    web_search_provider: str = "auto"
    # anthropic-only: model and native tool revision for server-side ``web_search``.
    web_search_model: str = "claude-haiku-4-5"
    web_search_tool_version: str = "web_search_20260209"

    # ===== Per-provider rate limiting =====
    # Token-bucket parameters used to throttle outbound LLM requests
    # proactively so we don't trip the provider's per-minute limits.
    anthropic_requests_per_second: float = 0.4   # ~24 RPM, fits 30k input-token/min tier
    anthropic_burst: int = 1
    openai_requests_per_second: float = 1.0      # 60 RPM
    openai_burst: int = 2

    # Inline retry policy for transient LLM errors (429, 5xx, network).
    llm_max_retries: int = 4
    llm_retry_initial_delay: float = 2.0
    llm_retry_max_delay: float = 60.0

    # Max output tokens per chat completion. Smaller cap reduces the
    # ``max_tokens`` we promise the provider, which (for Anthropic) reduces
    # the rate-limit accounting on each call.
    max_output_tokens: int = 2048

    # Checkpoint list polling: GET /api/agents/checkpoints is expensive (compiles
    # LangGraph + walks history). Short TTL avoids stacked duplicate work while
    # the frontend polls.
    checkpoints_cache_ttl_seconds: float = 15.0

    # Log each LLM completion call at INFO with route (slug + HTTP target hint).
    llm_invoke_log_each_call: bool = True
    # Set true to enable DEBUG logs from openai / httpcore (very noisy).
    llm_openai_http_debug: bool = False

    # Database
    database_url: str = "postgresql+psycopg://tdd:tdd_dev_password@localhost:5432/tdd_agentic"
    checkpointer_url: str = "postgresql://tdd:tdd_dev_password@localhost:5432/tdd_agentic"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Embeddings
    embedding_provider: str = Field(default="openai")  # "openai" or "local"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Tools
    tavily_api_key: str = ""
    fs_read_max_bytes: int = 32_768
    shell_output_max_chars: int = 8_000

    # Agent context budgets
    checkpoint_max_human_messages: int = 12
    skill_inject_max_chars: int = 2_000
    rag_default_k: int = 4
    rag_grade_cache_ttl_seconds: float = 300.0

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3001"

    # API
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_cors_origins: str = "http://localhost:3000"

    # Workspace where generated project artifacts live
    workspace_root: Path = Path("/app/workspace")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.backend_cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
