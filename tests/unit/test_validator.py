# Purpose: Unit tests for the validator/materializer.

from __future__ import annotations

import numpy as np

from shl_recommender.assembly.validator import MAX_RECOMMENDATIONS, materialize, validate_ids
from shl_recommender.catalog.loader import CatalogIndex, build_bm25
from shl_recommender.catalog.normalize import CatalogItem
from shl_recommender.catalog.normalize import build_search_text
from shl_recommender.catalog.retrieval import CategoryCoverage, l2_normalize


def _make_index(ids_and_names: list[tuple[str, str, str]]) -> CatalogIndex:
    items = [
        CatalogItem(
            entity_id=eid,
            name=name,
            url=f"https://www.shl.com/{eid}",
            description="",
            keys=("Knowledge & Skills",),
            test_type=tt,
            job_levels=(),
            languages=(),
            duration="",
            duration_minutes=None,
            remote=False,
            adaptive=False,
        )
        for eid, name, tt in ids_and_names
    ]
    bm25 = build_bm25(items, build_search_text)
    rng = np.random.default_rng(0)
    embeddings = l2_normalize(rng.standard_normal((len(items), 8)).astype(np.float32))
    coverage = CategoryCoverage(exemplars={letter: () for letter in "KPASBCDE"})
    return CatalogIndex(items=items, bm25=bm25, embeddings=embeddings, coverage=coverage)


def test_validate_drops_unknown_ids() -> None:
    idx = _make_index([("1", "A", "K"), ("2", "B", "P")])
    out = validate_ids(["1", "999", "2", "fake"], idx)
    assert out == ["1", "2"]


def test_validate_dedupes_preserving_order() -> None:
    idx = _make_index([("1", "A", "K"), ("2", "B", "P")])
    out = validate_ids(["2", "1", "2", "1"], idx)
    assert out == ["2", "1"]


def test_materialize_produces_valid_recommendations() -> None:
    idx = _make_index([("1", "Java Test", "K"), ("2", "OPQ", "P")])
    recs = materialize(["1", "2"], idx)
    assert len(recs) == 2
    assert recs[0].name == "Java Test"
    assert recs[0].url == "https://www.shl.com/1"
    assert recs[0].test_type == "K"


def test_materialize_caps_at_max() -> None:
    pairs = [(str(i), f"name-{i}", "K") for i in range(15)]
    idx = _make_index(pairs)
    recs = materialize([str(i) for i in range(15)], idx)
    assert len(recs) == MAX_RECOMMENDATIONS


def test_materialize_silently_skips_invalid_ids() -> None:
    idx = _make_index([("1", "A", "K")])
    recs = materialize(["999", "1", "fake"], idx)
    assert len(recs) == 1
    assert recs[0].name == "A"
