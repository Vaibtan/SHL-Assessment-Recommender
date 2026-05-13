# Purpose: Runtime configuration loaded from environment variables.

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Final


@dataclass(frozen=True, slots=True)
class Settings:
    """Frozen runtime configuration. Read once via `get_settings()`."""

    router_model: str
    handler_model: str
    embedding_model: str
    embedding_dims: int

    router_temperature: float
    recommend_temperature: float
    refine_temperature: float
    compare_temperature: float
    clarify_temperature: float
    top_p: float

    embedding_batch_size: int

    llm_timeout_seconds: float
    request_timeout_seconds: float



_DEFAULT_ROUTER_MODEL: Final[str] = "gemini-2.5-flash"
_DEFAULT_HANDLER_MODEL: Final[str] = "gemini-2.5-flash"
_DEFAULT_EMBEDDING_MODEL: Final[str] = "gemini-embedding-001"
_DEFAULT_EMBEDDING_DIMS: Final[int] = 768
_DEFAULT_ROUTER_TEMPERATURE: Final[float] = 0.0
_DEFAULT_RECOMMEND_TEMPERATURE: Final[float] = 0.1
_DEFAULT_REFINE_TEMPERATURE: Final[float] = 0.1
_DEFAULT_COMPARE_TEMPERATURE: Final[float] = 0.2
_DEFAULT_CLARIFY_TEMPERATURE: Final[float] = 0.3
_DEFAULT_TOP_P: Final[float] = 0.95
_DEFAULT_EMBEDDING_BATCH_SIZE: Final[int] = 32
_DEFAULT_LLM_TIMEOUT_SECONDS: Final[float] = 10.0
_DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[float] = 28.0




def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default




@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance.

    The first call at runtime materializes the Settings from the current env;
    subsequent calls return the same instance. Use `reset_settings_cache()` in
    tests if you need to re-read after manipulating env vars.
    """
    return Settings(
        router_model=_env_str("SHL_ROUTER_MODEL", _DEFAULT_ROUTER_MODEL),
        handler_model=_env_str("SHL_HANDLER_MODEL", _DEFAULT_HANDLER_MODEL),
        embedding_model=_env_str("SHL_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
        embedding_dims=_env_int("SHL_EMBEDDING_DIMS", _DEFAULT_EMBEDDING_DIMS),
        router_temperature=_env_float("SHL_ROUTER_TEMPERATURE", _DEFAULT_ROUTER_TEMPERATURE),
        recommend_temperature=_env_float(
            "SHL_RECOMMEND_TEMPERATURE", _DEFAULT_RECOMMEND_TEMPERATURE
        ),
        refine_temperature=_env_float("SHL_REFINE_TEMPERATURE", _DEFAULT_REFINE_TEMPERATURE),
        compare_temperature=_env_float("SHL_COMPARE_TEMPERATURE", _DEFAULT_COMPARE_TEMPERATURE),
        clarify_temperature=_env_float("SHL_CLARIFY_TEMPERATURE", _DEFAULT_CLARIFY_TEMPERATURE),
        top_p=_env_float("SHL_TOP_P", _DEFAULT_TOP_P),
        embedding_batch_size=_env_int(
            "SHL_EMBEDDING_BATCH_SIZE", _DEFAULT_EMBEDDING_BATCH_SIZE
        ),
        llm_timeout_seconds=_env_float(
            "SHL_LLM_TIMEOUT_SECONDS", _DEFAULT_LLM_TIMEOUT_SECONDS
        ),
        request_timeout_seconds=_env_float(
            "SHL_REQUEST_TIMEOUT_SECONDS", _DEFAULT_REQUEST_TIMEOUT_SECONDS
        ),
    )


def reset_settings_cache() -> None:
    """Clear the cached Settings — primarily useful for tests."""
    get_settings.cache_clear()




def __getattr__(name: str) -> object:
    settings = get_settings()
    mapping = {
        "ROUTER_MODEL": settings.router_model,
        "HANDLER_MODEL": settings.handler_model,
        "EMBEDDING_MODEL": settings.embedding_model,
        "EMBEDDING_DIMS": settings.embedding_dims,
    }
    if name in mapping:
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
