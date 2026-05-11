"""Unit tests for retrieval primitives — RRF, filters, hybrid, coverage."""

from __future__ import annotations

import numpy as np
import pytest

from shl_recommender.catalog.normalize import CatalogItem, build_search_text
from shl_recommender.catalog.retrieval import (
    CategoryCoverage,
    RetrievalFilters,
    Retriever,
    cosine_topk,
    l2_normalize,
    reciprocal_rank_fusion,
    tokenize,
)
from shl_recommender.catalog.loader import build_bm25


def _make_item(eid: str, name: str, test_type: str = "K", **kwargs) -> CatalogItem:
    return CatalogItem(
        entity_id=eid,
        name=name,
        url=f"https://www.shl.com/{eid}",
        description=kwargs.get("description", ""),
        keys=tuple(kwargs.get("keys", ["Knowledge & Skills"])),
        test_type=test_type,
        job_levels=tuple(kwargs.get("job_levels", [])),
        languages=tuple(kwargs.get("languages", [])),
        duration=kwargs.get("duration", ""),
        duration_minutes=kwargs.get("duration_minutes"),
        remote=kwargs.get("remote", False),
        adaptive=kwargs.get("adaptive", False),
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("C++ programming", ["c++", "programming"]),
        (".NET 4.5", [".net", "4.5"]),
        ("Verify G+ test", ["verify", "g+", "test"]),
        ("OPQ32r", ["opq32r"]),
        ("", []),
    ],
)
def test_tokenize_preserves_domain_chars(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


def test_rrf_fuses_with_known_scores() -> None:
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = reciprocal_rank_fusion([a, b], k=60)
    fused_dict = dict(fused)
    # x: 1/(60+1) + 1/(60+2) ; y: 1/(60+2) + 1/(60+1) -> tie. x and y both top.
    assert fused[0][0] in {"x", "y"}
    assert pytest.approx(fused_dict["x"], rel=1e-6) == pytest.approx(fused_dict["y"], rel=1e-6)
    # w (rank-2 in only one list) ranks below z (rank-2 in only one list) -> tie also.
    assert fused_dict["w"] == pytest.approx(fused_dict["z"], rel=1e-6)


def test_l2_normalize_unit_vectors() -> None:
    m = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    out = l2_normalize(m)
    assert pytest.approx(np.linalg.norm(out[0]), rel=1e-6) == 1.0
    # zero vector stays zero (divisor clamped, not divided by zero)
    assert np.all(np.isfinite(out[1]))


def test_cosine_topk_returns_descending() -> None:
    matrix = l2_normalize(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32))
    q = l2_normalize(np.array([[1.0, 0.0]], dtype=np.float32))[0]
    res = cosine_topk(q, matrix, k=2)
    assert [i for i, _ in res] == [0, 1]
    assert res[0][1] >= res[1][1]


def _build_retriever(items: list[CatalogItem]) -> Retriever:
    bm25 = build_bm25(items, build_search_text)
    rng = np.random.default_rng(0)
    embeddings = l2_normalize(rng.standard_normal((len(items), 8)).astype(np.float32))
    coverage = CategoryCoverage(exemplars={letter: () for letter in "KPASBCDE"})
    return Retriever(items, bm25, embeddings, coverage)


def test_hybrid_retrieve_returns_relevant_items() -> None:
    items = [
        _make_item("1", "Java 8 Test", description="Tests Java 8 features"),
        _make_item("2", "Python (New)", description="Tests Python knowledge"),
        _make_item("3", "Core Java Advanced", description="Advanced Java for senior IC"),
        _make_item("4", "Personality OPQ32r", test_type="P", description="Personality questionnaire"),
    ]
    r = _build_retriever(items)
    hits = r.retrieve("Java backend developer", final_k=3)
    assert len(hits) > 0
    ranked_ids = [h.entity_id for h in hits[:2]]
    # Java items should rank above the Python item.
    assert "1" in ranked_ids or "3" in ranked_ids


def test_hard_filter_drops_non_matching() -> None:
    items = [
        _make_item("1", "Java", test_type="K", duration_minutes=30),
        _make_item("2", "Quick Java", test_type="K", duration_minutes=5),
        _make_item("3", "Long Java", test_type="K", duration_minutes=60),
    ]
    r = _build_retriever(items)
    hits = r.retrieve("Java", filters=RetrievalFilters(duration_max_minutes=10))
    assert {h.entity_id for h in hits} == {"2"}


def test_duration_filter_drops_unknown_duration() -> None:
    items = [
        _make_item("1", "Known Quick Java", test_type="K", duration_minutes=8),
        _make_item("2", "Unknown Duration Java", test_type="K", duration_minutes=None),
    ]
    r = _build_retriever(items)
    hits = r.retrieve("Java", filters=RetrievalFilters(duration_max_minutes=10))
    assert {h.entity_id for h in hits} == {"1"}


def test_test_type_filter_anyof() -> None:
    items = [
        _make_item("1", "K item", test_type="K"),
        _make_item("2", "P item", test_type="P"),
        _make_item("3", "KP item", test_type="K,P"),
    ]
    r = _build_retriever(items)
    hits = r.retrieve("item", filters=RetrievalFilters(test_types=("P",)))
    ids = {h.entity_id for h in hits}
    assert "2" in ids and "3" in ids
    assert "1" not in ids


def test_category_coverage_injects_when_missing() -> None:
    items = [
        _make_item("1", "Java Knowledge", test_type="K", description="Java programming"),
        _make_item("2", "Python Knowledge", test_type="K", description="Python programming"),
        _make_item("99", "OPQ32r", test_type="P", description="Personality questionnaire"),
    ]
    bm25 = build_bm25(items, build_search_text)
    rng = np.random.default_rng(0)
    embeddings = l2_normalize(rng.standard_normal((len(items), 8)).astype(np.float32))
    coverage = CategoryCoverage(exemplars={letter: () for letter in "KPASBCDE"} | {"P": ("99",)})
    r = Retriever(items, bm25, embeddings, coverage)
    # Constrain retrieval so "99" doesn't get pulled in naturally:
    # query_vec=None disables dense retrieval, per_retriever_k=1 keeps only the top BM25 hit.
    hits = r.retrieve("Java", query_vec=None, coverage_letters=("P",), per_retriever_k=1)
    assert any(h.entity_id == "99" and h.injected for h in hits)


def test_category_coverage_does_not_double_inject() -> None:
    items = [
        _make_item("1", "Java", test_type="K"),
        _make_item("2", "OPQ32r", test_type="P", description="java relevant"),
    ]
    bm25 = build_bm25(items, build_search_text)
    rng = np.random.default_rng(0)
    embeddings = l2_normalize(rng.standard_normal((len(items), 8)).astype(np.float32))
    coverage = CategoryCoverage(exemplars={letter: () for letter in "KPASBCDE"} | {"P": ("2",)})
    r = Retriever(items, bm25, embeddings, coverage)
    hits = r.retrieve("java", coverage_letters=("P",))
    injected_count = sum(1 for h in hits if h.injected)
    assert injected_count == 0  # already present via retrieval


def test_find_similar_excludes_source() -> None:
    items = [_make_item(str(i), f"item {i}") for i in range(5)]
    r = _build_retriever(items)
    hits = r.find_similar("0", k=3)
    assert all(h.entity_id != "0" for h in hits)
