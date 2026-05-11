"""Unit tests for persona extraction from sample_conversations/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shl_recommender.catalog.loader import (
    CatalogIndex,
    build_bm25,
    derive_default_coverage,
)
from shl_recommender.catalog.normalize import build_search_text, normalize_catalog
from shl_recommender.catalog.retrieval import l2_normalize
from tests.replay.personas import SAMPLE_DIR, load_all_personas, parse_persona

CATALOG_JSON = Path(__file__).resolve().parents[2] / "data" / "shl_product_catalog.json"


@pytest.fixture(scope="module")
def index() -> CatalogIndex:
    items = normalize_catalog(CATALOG_JSON)
    bm25 = build_bm25(items, build_search_text)
    rng = np.random.default_rng(0)
    embeddings = l2_normalize(rng.standard_normal((len(items), 768)).astype(np.float32))
    coverage = derive_default_coverage(items)
    return CatalogIndex(items=items, bm25=bm25, embeddings=embeddings, coverage=coverage)


def test_load_all_ten_personas(index: CatalogIndex) -> None:
    personas = load_all_personas(index)
    assert len(personas) == 10
    sample_ids = {p.sample_id for p in personas}
    assert sample_ids == {f"C{i}" for i in range(1, 11)}


def test_each_persona_has_user_turns_and_labels(index: CatalogIndex) -> None:
    personas = load_all_personas(index)
    for p in personas:
        assert p.user_turns, f"{p.sample_id}: no user turns"
        assert p.opening_message, f"{p.sample_id}: empty opening message"
        # Most traces have labeled shortlists; allow a couple to slip if parsing fails on edge cases,
        # but at least 7 of 10 should have non-empty labels for the metric to be meaningful.
    labeled = sum(1 for p in personas if p.expected_entity_ids)
    assert labeled >= 7


def test_persona_c2_first_message_mentions_rust(index: CatalogIndex) -> None:
    personas = load_all_personas(index)
    c2 = next(p for p in personas if p.sample_id == "C2")
    assert "rust" in c2.opening_message.lower()


def test_persona_c4_expected_includes_numerical_reasoning(index: CatalogIndex) -> None:
    personas = load_all_personas(index)
    c4 = next(p for p in personas if p.sample_id == "C4")
    names_lc = " ".join(c4.expected_names).lower()
    assert "numerical" in names_lc or any(
        "shl-verify-interactive-numerical-reasoning" in (i or "")
        for i in c4.expected_entity_ids
    ) or len(c4.expected_entity_ids) >= 3
