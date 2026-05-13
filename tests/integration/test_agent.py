# Purpose: End-to-end integration tests with a fake LLM.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shl_recommender.agent.runner import Agent
from shl_recommender.catalog.loader import (
    build_bm25,
    derive_default_coverage,
    load_index,
)
from shl_recommender.catalog.normalize import build_search_text, normalize_catalog
from shl_recommender.catalog.retrieval import l2_normalize
from shl_recommender.schemas import ChatRequest, Message
from tests.integration.fakes import FakeLLMClient, stub_json, stub_text

CATALOG_JSON = Path(__file__).resolve().parents[2] / "data" / "shl_product_catalog.json"


@pytest.fixture(scope="module")
def index():
    """Build an in-memory index with deterministic placeholder embeddings."""
    items = normalize_catalog(CATALOG_JSON)
    bm25 = build_bm25(items, build_search_text)
    rng = np.random.default_rng(42)
    embeddings = l2_normalize(rng.standard_normal((len(items), 768)).astype(np.float32))
    coverage = derive_default_coverage(items)
    from shl_recommender.catalog.loader import CatalogIndex

    return CatalogIndex(items=items, bm25=bm25, embeddings=embeddings, coverage=coverage)


@pytest.mark.asyncio
async def test_clarify_intent_for_vague_query(index) -> None:
    llm = FakeLLMClient()
    llm.router_replies.append(
        stub_json(
            {
                "intent": "clarify",
                "clarifying_question": "What role are you hiring for, and at what seniority?",
                "is_final_turn": False,
            }
        )
    )
    agent = Agent(index=index, llm=llm)
    req = ChatRequest(messages=[Message(role="user", content="I need an assessment")])
    out = await agent.chat(req)
    assert out.decision.intent.value == "clarify"
    assert out.response.recommendations == []
    assert out.response.end_of_conversation is False
    assert "?" in out.response.reply


@pytest.mark.asyncio
async def test_refuse_intent_for_off_topic(index) -> None:
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
    agent = Agent(index=index, llm=llm)
    req = ChatRequest(messages=[Message(role="user", content="What's the weather in Bangalore?")])
    out = await agent.chat(req)
    assert out.decision.intent.value == "refuse"
    assert out.response.end_of_conversation is False
    assert "scope" in out.response.reply.lower() or "selection" in out.response.reply.lower()


@pytest.mark.asyncio
async def test_recommend_returns_valid_shortlist(index) -> None:
    llm = FakeLLMClient()
    llm.router_replies.append(
        stub_json(
            {
                "intent": "recommend",
                "search_query": "senior Java backend engineer Spring SQL AWS",
                "filters": {},
                "coverage_letters": ["P", "A"],
                "is_final_turn": False,
            }
        )
    )
    java_id = next(it.entity_id for it in index.items if "core java" in it.name.lower())
    opq_id = next(it.entity_id for it in index.items if "opq32r" in it.name.lower())
    verify_id = next(it.entity_id for it in index.items if "verify interactive g+" in it.name.lower())
    llm.handler_replies.append(
        stub_json(
            {
                "entity_ids": [java_id, opq_id, verify_id],
                "reply": "Three core picks for a senior Java IC.",
            }
        )
    )

    agent = Agent(index=index, llm=llm)
    req = ChatRequest(
        messages=[
            Message(role="user", content="Hiring senior Java backend engineer, 5 years Spring/SQL/AWS"),
        ]
    )
    out = await agent.chat(req)
    assert out.decision.intent.value == "recommend"
    assert 1 <= len(out.response.recommendations) <= 10
    rec_urls = {r.url for r in out.response.recommendations}
    assert all(url.startswith("https://www.shl.com/") for url in rec_urls)
    assert "| # | Name |" in out.response.reply
    assert out.response.end_of_conversation is False


@pytest.mark.asyncio
async def test_recommend_drops_hallucinated_ids(index) -> None:
    llm = FakeLLMClient()
    llm.router_replies.append(
        stub_json(
            {
                "intent": "recommend",
                "search_query": "Java",
                "filters": {},
                "coverage_letters": [],
                "is_final_turn": False,
            }
        )
    )
    real_id = next(it.entity_id for it in index.items if "java" in it.name.lower())
    llm.handler_replies.append(
        stub_json({"entity_ids": ["FAKE-ID-1", real_id, "ANOTHER-FAKE"], "reply": "ok"})
    )
    agent = Agent(index=index, llm=llm)
    req = ChatRequest(messages=[Message(role="user", content="Hiring Java engineer")])
    out = await agent.chat(req)
    rec_urls = {r.url for r in out.response.recommendations}
    assert all(url.startswith("https://www.shl.com/") for url in rec_urls)
    assert any(r.url == next(it.url for it in index.items if it.entity_id == real_id) for r in out.response.recommendations)
    assert all("FAKE" not in r.name for r in out.response.recommendations)
    assert out.handler_result.validation_errors == ["dropped_invalid_or_out_of_pool_ids:2"]


@pytest.mark.asyncio
async def test_compare_uses_grounded_records(index) -> None:
    llm = FakeLLMClient()
    llm.router_replies.append(
        stub_json(
            {
                "intent": "compare",
                "compare_pair": {
                    "a": "Occupational Personality Questionnaire OPQ32r",
                    "b": "Dependability and Safety Instrument (DSI)",
                },
                "is_final_turn": False,
            }
        )
    )
    llm.handler_replies.append(
        stub_text(
            "OPQ32r is a broad personality questionnaire. DSI focuses specifically on safety-relevant traits."
        )
    )
    agent = Agent(index=index, llm=llm)
    req = ChatRequest(
        messages=[Message(role="user", content="What's the difference between OPQ and DSI?")]
    )
    out = await agent.chat(req)
    assert out.decision.intent.value == "compare"
    assert "OPQ32r" in out.response.reply or "personality" in out.response.reply.lower()
    assert out.response.end_of_conversation is False


@pytest.mark.asyncio
async def test_refine_with_confirmation_ends_conversation(index) -> None:
    llm = FakeLLMClient()
    java = next(it for it in index.items if "core java" in it.name.lower())
    opq = next(it for it in index.items if "opq32r" in it.name.lower())
    prior_md = (
        "Here's the shortlist:\n\n"
        "| # | Name | URL |\n|---|---|---|\n"
        f"| 1 | {java.name} | <{java.url}> |\n"
        f"| 2 | {opq.name} | <{opq.url}> |"
    )
    llm.router_replies.append(
        stub_json(
            {
                "intent": "refine",
                "search_query": "",
                "constraint_deltas": {"add": [], "drop": [], "swap": []},
                "is_final_turn": True,
            }
        )
    )
    agent = Agent(index=index, llm=llm)
    req = ChatRequest(
        messages=[
            Message(role="user", content="Hiring Java engineer"),
            Message(role="assistant", content=prior_md),
            Message(role="user", content="Perfect, locking it in"),
        ]
    )
    out = await agent.chat(req)
    assert out.decision.intent.value == "refine"
    assert out.response.end_of_conversation is True
    assert {r.url for r in out.response.recommendations} >= {java.url, opq.url}


@pytest.mark.asyncio
async def test_refine_drop_removes_item(index) -> None:
    llm = FakeLLMClient()
    java = next(it for it in index.items if "core java" in it.name.lower())
    opq = next(it for it in index.items if "opq32r" in it.name.lower())
    prior_md = (
        "| # | Name | URL |\n|---|---|---|\n"
        f"| 1 | {java.name} | <{java.url}> |\n"
        f"| 2 | {opq.name} | <{opq.url}> |"
    )
    llm.router_replies.append(
        stub_json(
            {
                "intent": "refine",
                "search_query": "Java backend without personality",
                "constraint_deltas": {"add": [], "drop": ["personality"], "swap": []},
                "is_final_turn": False,
            }
        )
    )
    llm.handler_replies.append(
        stub_json({"entity_ids": [java.entity_id], "reply": "Dropped OPQ."})
    )
    agent = Agent(index=index, llm=llm)
    req = ChatRequest(
        messages=[
            Message(role="user", content="Hiring Java engineer"),
            Message(role="assistant", content=prior_md),
            Message(role="user", content="Drop the personality one"),
        ]
    )
    out = await agent.chat(req)
    rec_urls = {r.url for r in out.response.recommendations}
    assert java.url in rec_urls
    assert opq.url not in rec_urls
    assert out.response.end_of_conversation is False
