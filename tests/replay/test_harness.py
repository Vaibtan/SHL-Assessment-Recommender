"""Replay harness smoke test using a scripted FakeLLMClient.

Confirms the harness shape (turn loop, schema validation, recall computation)
works end-to-end without making network calls. Live LLM replay runs are driven
by the `replay_live.py` script, not by pytest.
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
from tests.integration.fakes import FakeLLMClient, stub_json
from tests.replay.harness import replay_persona
from tests.replay.personas import load_all_personas

CATALOG_JSON = Path(__file__).resolve().parents[2] / "data" / "shl_product_catalog.json"


@pytest.fixture(scope="module")
def index() -> CatalogIndex:
    items = normalize_catalog(CATALOG_JSON)
    bm25 = build_bm25(items, build_search_text)
    rng = np.random.default_rng(0)
    embeddings = l2_normalize(rng.standard_normal((len(items), 768)).astype(np.float32))
    coverage = derive_default_coverage(items)
    return CatalogIndex(items=items, bm25=bm25, embeddings=embeddings, coverage=coverage)


@pytest.mark.asyncio
async def test_harness_completes_a_persona_with_scripted_llm(index: CatalogIndex) -> None:
    """Drive C1 (senior leadership) through the harness with a fake LLM."""
    personas = load_all_personas(index)
    c1 = next(p for p in personas if p.sample_id == "C1")
    assert c1.expected_entity_ids, "C1 should have labeled shortlist"

    llm = FakeLLMClient()

    # Turn 1: vague — clarify
    llm.router_replies.append(
        stub_json(
            {
                "intent": "clarify",
                "clarifying_question": "Who is this meant for?",
                "is_final_turn": False,
            }
        )
    )
    # Turn 2: more context — recommend
    llm.router_replies.append(
        stub_json(
            {
                "intent": "recommend",
                "search_query": "senior leadership executive directors CXO 15+ years",
                "filters": {},
                "coverage_letters": ["P"],
                "is_final_turn": False,
            }
        )
    )
    llm.handler_replies.append(
        stub_json(
            {
                "entity_ids": list(c1.expected_entity_ids[:3]),
                "reply": "OPQ32r and the leadership reports.",
            }
        )
    )

    agent = Agent(index=index, llm=llm)
    record = await replay_persona(c1, agent, max_turns=8)

    assert record.schema_valid is True
    assert record.final_predicted_ids
    # Recall@10 should be > 0 since we explicitly returned overlapping ids.
    assert record.recall_at_10 > 0.0
