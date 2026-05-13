"""Application configuration loaded from environment via pydantic-settings."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_logger = logging.getLogger(__name__)


def _default_workspace_root() -> Path:
    docker_root = Path("/app/workspace")
    if docker_root.is_dir():
        return docker_root
    local_root = Path.cwd() / "workspace"
    local_root.mkdir(parents=True, exist_ok=True)
    return local_root


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
    # Researcher-only OpenAI HTTP host when ``researcher_model`` is ``openai/...``.
    # Use when ``OPENAI_BASE_URL`` points at a local server but the researcher should
    # call OpenAI's cloud API. Non-empty ``RESEARCHER_OPENAI_BASE_URL`` wins over the flag.
    researcher_openai_base_url: str = ""
    researcher_use_platform_openai: bool = False
    # Grader-only OpenAI HTTP host when ``grader_model`` is ``openai/...`` (same pattern as
    # the researcher). Non-empty ``GRADER_OPENAI_BASE_URL`` wins over the flag.
    grader_openai_base_url: str = ""
    grader_use_platform_openai: bool = False
    # Per-role: set true when that role's ``openai/...`` traffic goes to a local or
    # non-standard OpenAI-compatible server (e.g. llama.cpp) where ``tool_choice`` and
    # tool schemas may differ. Leave false when ``OPENAI_BASE_URL`` points at the real
    # OpenAI API or another fully compatible host.
    pm_is_local: bool = False
    researcher_is_local: bool = False
    lead_is_local: bool = False
    dev_is_local: bool = False
    backend_dev_is_local: bool = False
    frontend_dev_is_local: bool = False
    grader_is_local: bool = False
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
    workspace_root: Path = Field(default_factory=_default_workspace_root)

    # LangChain Deep Agents (specialist harness). Set false to use the legacy
    # tool-loop subgraph for the researcher only.
    use_deep_agent_researcher: bool = True
    # Max LangGraph steps for one specialist deep-agent invocation (inner loop).
    deep_agent_recursion_limit: int = 80

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.backend_cors_origins.split(",") if o.strip()]

    @field_validator("workspace_root", mode="before")
    @classmethod
    def _coerce_workspace_root(cls, value: object) -> Path:
        if value is None or (isinstance(value, str) and not value.strip()):
            return _default_workspace_root()
        return Path(value)

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _strip_openai_api_key(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


def log_settings_warnings(settings: Settings | None = None) -> None:
    """Emit configuration warnings once at startup."""
    settings = settings or get_settings()
    db_host = urlparse(settings.database_url.replace("+psycopg", "")).hostname
    ck_host = urlparse(settings.checkpointer_url).hostname
    if db_host and ck_host and db_host != ck_host:
        _logger.warning(
            "DATABASE_URL host %r differs from CHECKPOINTER_URL host %r; "
            "tickets and LangGraph checkpoints may be on different databases.",
            db_host,
            ck_host,
        )
    provider = (settings.web_search_provider or "auto").strip().lower()
    has_tavily = bool(settings.tavily_api_key.strip())
    has_anthropic = bool(settings.anthropic_api_key.strip())
    if provider == "tavily" and not has_tavily:
        _logger.warning("WEB_SEARCH_PROVIDER=tavily but TAVILY_API_KEY is empty.")
    elif provider == "auto" and not has_tavily and not has_anthropic:
        _logger.warning(
            "Researcher web_search has no backend: set TAVILY_API_KEY and/or ANTHROPIC_API_KEY."
        )
    if settings.researcher_model.lower().startswith("openai/"):
        r_url = (settings.researcher_openai_base_url or "").strip()
        uses_researcher_cloud = bool(r_url or settings.researcher_use_platform_openai)
        if settings.openai_base_url.strip() and not uses_researcher_cloud:
            if settings.researcher_is_local:
                _logger.warning(
                    "Researcher model %s uses OPENAI_BASE_URL with RESEARCHER_IS_LOCAL=true; "
                    "first-hop OpenAI tool_choice forcing is disabled for compatibility.",
                    settings.researcher_model,
                )
            else:
                _logger.info(
                    "Researcher model %s inherits OPENAI_BASE_URL (same host as other openai/ roles). "
                    "For cloud OpenAI while keeping a local OPENAI_BASE_URL for others, set "
                    "RESEARCHER_USE_PLATFORM_OPENAI=true or RESEARCHER_OPENAI_BASE_URL=https://api.openai.com/v1.",
                    settings.researcher_model,
                )
        elif uses_researcher_cloud and settings.openai_base_url.strip():
            _logger.info(
                "Researcher OpenAI calls use RESEARCHER_OPENAI_BASE_URL / "
                "RESEARCHER_USE_PLATFORM_OPENAI, not OPENAI_BASE_URL.",
            )
    if settings.grader_model.lower().startswith("openai/"):
        g_url = (settings.grader_openai_base_url or "").strip()
        uses_grader_cloud = bool(g_url or settings.grader_use_platform_openai)
        if settings.openai_base_url.strip() and not uses_grader_cloud:
            _logger.info(
                "Grader model %s inherits OPENAI_BASE_URL (same host as other openai/ roles). "
                "For cloud OpenAI while keeping a local OPENAI_BASE_URL for others, set "
                "GRADER_USE_PLATFORM_OPENAI=true or GRADER_OPENAI_BASE_URL=https://api.openai.com/v1.",
                settings.grader_model,
            )
        elif uses_grader_cloud and settings.openai_base_url.strip():
            _logger.info(
                "Grader OpenAI calls use GRADER_OPENAI_BASE_URL / "
                "GRADER_USE_PLATFORM_OPENAI, not OPENAI_BASE_URL.",
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
