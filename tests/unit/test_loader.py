# Purpose: Unit tests for catalog index loading and lookup behavior.

from __future__ import annotations

import numpy as np
import pytest

from shl_recommender.catalog.loader import CatalogIndex, build_bm25, load_index, save_index
from shl_recommender.catalog.normalize import CatalogItem, build_search_text
from shl_recommender.catalog.retrieval import CategoryCoverage, l2_normalize
from shl_recommender.config import reset_settings_cache


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


def test_catalog_index_provides_cached_lookup_maps() -> None:
    idx = _make_index([("1", "Java Test", "K"), ("2", "OPQ32r", "P")])

    assert idx.get("1").name == "Java Test"
    assert idx.get_by_url("https://www.shl.com/1/").entity_id == "1"
    assert idx.get_by_name("opq32r").entity_id == "2"
    assert idx.items_for_ids(["2", "missing", "1"]) == [idx.get("2"), idx.get("1")]
    assert idx.retriever is idx.retriever


def test_catalog_index_resolves_fuzzy_names() -> None:
    idx = _make_index([("1", "Dependability and Safety Instrument (DSI)", "P")])

    assert idx.resolve_name("Dependability Safety Instrument").entity_id == "1"
    assert idx.suggest_name("Dependability Safety") == "Dependability and Safety Instrument (DSI)"


def test_load_index_rejects_embedding_dim_mismatch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    idx = _make_index([("1", "A", "K"), ("2", "B", "P")])
    save_index(tmp_path, idx.items, idx.embeddings, idx.bm25, idx.coverage)

    monkeypatch.setenv("SHL_EMBEDDING_DIMS", "768")
    reset_settings_cache()

    with pytest.raises(ValueError, match="SHL_EMBEDDING_DIMS"):
        load_index(tmp_path)

    reset_settings_cache()
