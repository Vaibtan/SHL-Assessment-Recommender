"""Unit tests for the env-driven config layer."""

from __future__ import annotations

import importlib

import pytest

from shl_recommender.config import get_settings, reset_settings_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with a fresh settings cache and ends restoring it."""
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_defaults_match_locked_design(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear any operator overrides so we observe the locked defaults.
    for key in (
        "SHL_ROUTER_MODEL",
        "SHL_HANDLER_MODEL",
        "SHL_EMBEDDING_MODEL",
        "SHL_EMBEDDING_DIMS",
        "SHL_ROUTER_TEMPERATURE",
        "SHL_RECOMMEND_TEMPERATURE",
        "SHL_REFINE_TEMPERATURE",
        "SHL_COMPARE_TEMPERATURE",
        "SHL_CLARIFY_TEMPERATURE",
        "SHL_TOP_P",
        "SHL_EMBEDDING_BATCH_SIZE",
        "SHL_LLM_TIMEOUT_SECONDS",
        "SHL_REQUEST_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    s = get_settings()
    assert s.router_model == "gemini-2.5-flash"
    assert s.handler_model == "gemini-2.5-flash"
    assert s.embedding_model == "gemini-embedding-001"
    assert s.embedding_dims == 768
    assert s.router_temperature == pytest.approx(0.0)
    assert s.recommend_temperature == pytest.approx(0.1)
    assert s.refine_temperature == pytest.approx(0.1)
    assert s.compare_temperature == pytest.approx(0.2)
    assert s.clarify_temperature == pytest.approx(0.3)
    assert s.top_p == pytest.approx(0.95)
    assert s.embedding_batch_size == 32
    assert s.llm_timeout_seconds == pytest.approx(10.0)
    assert s.request_timeout_seconds == pytest.approx(28.0)


def test_env_overrides_take_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHL_ROUTER_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("SHL_HANDLER_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("SHL_EMBEDDING_MODEL", "text-embedding-004")
    monkeypatch.setenv("SHL_EMBEDDING_DIMS", "256")
    monkeypatch.setenv("SHL_ROUTER_TEMPERATURE", "0.4")
    monkeypatch.setenv("SHL_TOP_P", "0.8")
    monkeypatch.setenv("SHL_EMBEDDING_BATCH_SIZE", "64")
    monkeypatch.setenv("SHL_LLM_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("SHL_REQUEST_TIMEOUT_SECONDS", "12.5")
    reset_settings_cache()
    s = get_settings()
    assert s.router_model == "gemini-2.5-pro"
    assert s.handler_model == "gemini-2.5-flash-lite"
    assert s.embedding_model == "text-embedding-004"
    assert s.embedding_dims == 256
    assert s.router_temperature == pytest.approx(0.4)
    assert s.top_p == pytest.approx(0.8)
    assert s.embedding_batch_size == 64
    assert s.llm_timeout_seconds == pytest.approx(3.5)
    assert s.request_timeout_seconds == pytest.approx(12.5)


def test_invalid_int_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHL_EMBEDDING_DIMS", "not-an-int")
    reset_settings_cache()
    s = get_settings()
    assert s.embedding_dims == 768  # default


def test_invalid_float_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHL_ROUTER_TEMPERATURE", "hot")
    reset_settings_cache()
    s = get_settings()
    assert s.router_temperature == pytest.approx(0.0)


def test_empty_string_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHL_ROUTER_MODEL", "   ")
    reset_settings_cache()
    s = get_settings()
    assert s.router_model == "gemini-2.5-flash"


def test_llm_module_resurfaces_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """`from shl_recommender.agent.llm import HANDLER_MODEL` still works."""
    monkeypatch.setenv("SHL_HANDLER_MODEL", "custom-model-x")
    reset_settings_cache()
    from shl_recommender.agent import llm as llm_module

    importlib.reload(llm_module)
    assert llm_module.HANDLER_MODEL == "custom-model-x"
