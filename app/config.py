"""Central configuration. Everything comes from .env (see .env.example).

Latency-relevant defaults are chosen for the <500 ms production target:
small output budgets, a hard LLM deadline, short in-process cache TTLs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- provider ---------------------------------------------------------
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    local_base_url: str = os.getenv("LOCAL_BASE_URL") or os.getenv("LOCAL_LLM_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    local_model: str = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
    local_api_key: str = os.getenv("LOCAL_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL") or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    openai_model: str = os.getenv("OPENAI_MODEL") or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    openai_api_key: str = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY", "")

    # --- LLM behaviour -------------------------------------------------------
    # json_schema  = constrained decoding (Ollama supports this)
    # json_object  = "any valid JSON" mode (fallback)
    # none         = plain generation + tolerant parser
    structured_mode: str = os.getenv("STRUCTURED_MODE", "json_schema").strip().lower()
    temperature: float = _float("TEMPERATURE", 0.0)
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 192)
    # Hard wall-clock cap for the LLM leg of a request. When exceeded the user gets a
    # deterministic fallback instead of a hung connection. Local Ollama on CPU/small GPU
    # needs more headroom than a cloud API - raise this if you see fallback replies.
    llm_deadline_ms: int = _int("LLM_DEADLINE_MS", 3000)
    llm_connect_timeout_s: float = _float("LLM_CONNECT_TIMEOUT_S", 2.0)
    llm_reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "").strip()
    # Small retry budget for transient network and upstream provider failures.
    llm_max_retries: int = _int("LLM_MAX_RETRIES", 1)
    # History sent to the LLM. The COMPLETE history is always kept in Redis; this only shapes what the model sees.
    #   last N turn pairs verbatim + older messages shortened to N chars + hard token cap (0 = no cap).
    #   Set LLM_HISTORY_RECENT_TURNS=100000 and LLM_HISTORY_MAX_TOKENS=0 to send everything verbatim.
    llm_history_recent_turns: int = _int("LLM_HISTORY_RECENT_TURNS", 6)
    llm_history_compact_chars: int = _int("LLM_HISTORY_COMPACT_CHARS", 160)
    llm_history_max_tokens: int = _int("LLM_HISTORY_MAX_TOKENS", 2000)
    llm_answer_cache: bool = _bool("LLM_ANSWER_CACHE", True)

    # --- summarisation settings ------------------------------------------------
    # Model for background history summarisation (a small, cheap LLM).
    llm_summary_model: str = os.getenv("LLM_SUMMARY_MODEL", "gpt-oss-20b")
    # Token budget for the generated summary (keep it short to stay cheap).
    llm_summary_tokens: int = _int("LLM_SUMMARY_TOKENS", 150)

    # --- router --------------------------------------------------------------
    router_enabled: bool = _bool("ROUTER_ENABLED", True)          # False = everything goes to the LLM (A/B baseline)

    # --- intent understanding: stage 2 between the rules and the LLM general path ------------------
    # Optional fast classifier settings retained for compatibility with deployments
    # that still enable the legacy intent stage.
    intent_router: str = os.getenv("INTENT_ROUTER", "on").strip().lower()          # on | off
    intent_llm_max_tokens: int = _int("INTENT_LLM_MAX_TOKENS", 20)
    intent_llm_timeout_ms: int = _int("INTENT_LLM_TIMEOUT_MS", 1500)
    intent_llm_max_chars: int = _int("INTENT_LLM_MAX_CHARS", 300)

    # --- SQL routing (text-to-SQL for analytical inventory queries) -------
    sql_routing: str = os.getenv("SQL_ROUTING", "on").strip().lower()           # on | off
    sql_llm_provider: str = os.getenv("SQL_LLM_PROVIDER", "").strip().lower()
    sql_llm_model: str = os.getenv("SQL_LLM_MODEL", "").strip()
    sql_llm_base_url: str = os.getenv("SQL_LLM_BASE_URL", "").strip()
    sql_timeout_ms: int = _int("SQL_TIMEOUT_MS", 120_000)          # local text-to-SQL can be slow - give it room

    # --- redis ---------------------------------------------------------------
    # Use REDIS_URL=fakeredis:// for a zero-dependency dev/test run (needs requirements-dev.txt).
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_timeout_ms: int = _int("REDIS_TIMEOUT_MS", 3000)
    redis_pool_size: int = _int("REDIS_POOL_SIZE", 50)
    session_ttl: int = _int("SESSION_TTL_SECONDS", 3600)
    history_max_turns: int = _int("HISTORY_MAX_TURNS", 12)

    # --- in-process (L1) cache ----------------------------------------------
    l1_ttl_s: int = _int("L1_TTL_SECONDS", 30)
    l1_max_entries: int = _int("L1_MAX_ENTRIES", 5000)

    # --- misc ----------------------------------------------------------------
    timezone: str = os.getenv("APP_TIMEZONE", "Asia/Kolkata")
    today_override: str = os.getenv("TODAY_OVERRIDE", "").strip()    # YYYY-MM-DD, tests/benchmarks only
    inventory_result_limit: int = _int("INVENTORY_RESULT_LIMIT", 5)
    database_url: str = os.getenv("DATABASE_URL", "").strip()        # optional Postgres write-behind
    rate_limit_enabled: bool = _bool("RATE_LIMIT_ENABLED", True)     # turn off for load tests / benchmarks
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    # Shared secret for /service/slots/push and /service/slots/seed (X-Webhook-Secret header). Empty = unchecked (dev only).
    service_webhook_secret: str = os.getenv("SERVICE_WEBHOOK_SECRET", "").strip()
    data_file: Path = field(default_factory=lambda: BASE_DIR / os.getenv("DATA_FILE", "data/customers.json"))

    # --- showroom capacity: max concurrent appointments of one type in the same date+time slot ---
    slot_capacity_test_drive: int = _int("SLOT_CAPACITY_TEST_DRIVE", 2)
    slot_capacity_default: int = _int("SLOT_CAPACITY_DEFAULT", 4)

    def slot_capacity(self, appointment_type: str) -> int:
        return self.slot_capacity_test_drive if appointment_type == "TEST_DRIVE" else self.slot_capacity_default

    @property
    def model_name(self) -> str:
        return {
            "ollama": self.ollama_model,
            "groq": self.groq_model,
            "local": self.local_model,
            "openai": self.openai_model,
        }.get(self.llm_provider, self.ollama_model)


settings = Settings()