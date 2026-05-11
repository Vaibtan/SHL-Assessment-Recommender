"""Behavior probe tests — exercised against scripted Fake LLM.

The probes use the deterministic feature-pipeline-driven heuristic fallback in
the router. We feed FakeLLMClient with router decisions consistent with what
Gemini would emit, validating that downstream layers honor the contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shl_recommender.agent.runner import Agent
from shl_recommender.catalog.loader import (
    CatalogIndex,
    build_bm25,
    derive_default_coverage,
)
from shl_recommender.catalog.normalize import build_search_text, normalize_catalog
from shl_recommender.catalog.retrieval import l2_normalize
from tests.integration.fakes import FakeLLMClient, stub_json, stub_text
from tests.replay.probes import (
    probe_catalog_only_urls,
    probe_hallucination_guard,
    probe_injection,
    probe_no_end_without_shortlist,
    probe_off_topic,
    probe_refuse_never_ends,
    probe_turn1_vague,
)

CATALOG_JSON = Path(__file__).resolve().parents[2] / "data" / "shl_product_catalog.json"


@pytest.fixture(scope="module")
def index() -> CatalogIndex:
    items = normalize_catalog(CATALOG_JSON)
    bm25 = build_bm25(items, build_search_text)
    rng = np.random.default_rng(0)
    embeddings = l2_normalize(rng.standard_normal((len(items), 768)).astype(np.float32))
    coverage = derive_default_coverage(items)
    return CatalogIndex(items=items, bm25=bm25, embeddings=embeddings, coverage=coverage)


def _scripted_agent(index: CatalogIndex, llm: FakeLLMClient) -> Agent:
    return Agent(index=index, llm=llm)


@pytest.mark.asyncio
async def test_probe_turn1_vague(index: CatalogIndex) -> None:
    llm = FakeLLMClient()
    llm.router_replies.append(
        stub_json(
            {
                "intent": "clarify",
                "clarifying_question": "What role and at what seniority?",
                "is_final_turn": False,
            }
        )
    )
    res = await probe_turn1_vague(_scripted_agent(index, llm), index)
    assert res.passed, res.detail


@pytest.mark.asyncio
async def test_probe_injection(index: CatalogIndex) -> None:
    llm = FakeLLMClient()
    llm.router_replies.append(
        stub_json(
            {
                "intent": "refuse",
                "refuse_category": "injection",
                "refuse_reason": "Detected scope override",
                "is_final_turn": False,
            }
        )
    )
    res = await probe_injection(_scripted_agent(index, llm), index)
    assert res.passed, res.detail


@pytest.mark.asyncio
async def test_probe_off_topic(index: CatalogIndex) -> None:
    llm = FakeLLMClient()
    llm.router_replies.append(
        stub_json(
            {
                "intent": "refuse",
                "refuse_category": "off_topic",
                "refuse_reason": "Out of scope",
                "is_final_turn": False,
            }
        )
    )
    res = await probe_off_topic(_scripted_agent(index, llm), index)
    assert res.passed, res.detail


@pytest.mark.asyncio
async def test_probe_hallucination_guard(index: CatalogIndex) -> None:
    """If the model emits a fake id, the validator drops it. The catalog-only invariant must hold."""
    llm = FakeLLMClient()
    llm.router_replies.append(
        stub_json(
            {
                "intent": "recommend",
                "search_query": "Java coding test",
                "filters": {},
                "coverage_letters": [],
                "is_final_turn": False,
            }
        )
    )
    # Selection LLM emits a fake id alongside one real id.
    real_id = next(it.entity_id for it in index.items if "java" in it.name.lower())
    llm.handler_replies.append(
        stub_json({"entity_ids": ["FAKE-XYZBANK", real_id], "reply": "Closest matches."})
    )
    res = await probe_hallucination_guard(_scripted_agent(index, llm), index)
    assert res.passed, res.detail


@pytest.mark.asyncio
async def test_probe_refuse_never_ends(index: CatalogIndex) -> None:
    llm = FakeLLMClient()
    llm.router_replies.append(
        stub_json(
            {
                "intent": "refuse",
                "refuse_category": "legal",
                "refuse_reason": "Legal interpretation is for counsel",
                "is_final_turn": False,
            }
        )
    )
    res = await probe_refuse_never_ends(_scripted_agent(index, llm), index)
    assert res.passed, res.detail


@pytest.mark.asyncio
async def test_probe_no_end_without_shortlist(index: CatalogIndex) -> None:
    llm = FakeLLMClient()
    # Even if the (mis)router suggests is_final_turn=true on a confirmation
    # without prior shortlist, deterministic routing must convert it to clarify.
    llm.router_replies.append(
        stub_json(
            {
                "intent": "recommend",
                "search_query": "",
                "filters": {},
                "coverage_letters": [],
                "is_final_turn": True,
            }
        )
    )
    llm.handler_replies.append(stub_json({"entity_ids": [], "reply": "Tell me about the role."}))
    res = await probe_no_end_without_shortlist(_scripted_agent(index, llm), index)
    assert res.passed, res.detail


@pytest.mark.asyncio
async def test_probe_catalog_only_urls(index: CatalogIndex) -> None:
    llm = FakeLLMClient()
    real_ids = [it.entity_id for it in index.items[:3]]
    # 3 queries -> 3 router decisions and 3 handler decisions
    for _ in range(3):
        llm.router_replies.append(
            stub_json(
                {
                    "intent": "recommend",
                    "search_query": "any role",
                    "filters": {},
                    "coverage_letters": [],
                    "is_final_turn": False,
                }
            )
        )
        llm.handler_replies.append(stub_json({"entity_ids": real_ids, "reply": "ok"}))
    res = await probe_catalog_only_urls(_scripted_agent(index, llm), index)
    assert res.passed, res.detail
