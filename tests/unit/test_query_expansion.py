from __future__ import annotations

from pathlib import Path

import pytest

from shl_recommender.catalog.normalize import CatalogItem, normalize_catalog
from shl_recommender.catalog.query_expansion import expand_catalog_candidates

CATALOG_JSON = Path(__file__).resolve().parents[2] / "data" / "shl_product_catalog.json"


@pytest.fixture(scope="module")
def catalog_items() -> list[CatalogItem]:
    return normalize_catalog(CATALOG_JSON)


def test_expands_exact_software_stack_terms(catalog_items: list[CatalogItem]) -> None:
    expanded = expand_catalog_candidates(
        "Senior full-stack engineer using Core Java, Spring, SQL, AWS, and Docker.",
        catalog_items,
    )

    names = {item.name for item in expanded.items}
    assert any("Core Java" in name for name in names)
    assert any(name.startswith("Spring") for name in names)
    assert any(name.startswith("SQL") for name in names)
    assert any("Amazon Web Services (AWS) Development" in name for name in names)
    assert any(name.startswith("Docker") for name in names)
    assert "software_engineering_stack" in expanded.matched_concepts


def test_expands_acronyms_from_catalog_names(catalog_items: list[CatalogItem]) -> None:
    expanded = expand_catalog_candidates(
        "Need a shortlist with OPQ, DSI, HIPAA, and G+ coverage.",
        catalog_items,
    )

    names = {item.name for item in expanded.items}
    assert any("OPQ32r" in name for name in names)
    assert any("Dependability and Safety Instrument (DSI)" in name for name in names)
    assert any(name.startswith("HIPAA") for name in names)
    assert any("SHL Verify Interactive G+" in name for name in names)


def test_expands_domain_synonyms_without_sample_specific_ids(catalog_items: list[CatalogItem]) -> None:
    expanded = expand_catalog_candidates(
        "We need Microsoft Office screening for administrative assistants: Excel and Word.",
        catalog_items,
    )

    names = {item.name for item in expanded.items}
    assert any(name.startswith("Microsoft 365") for name in names)
    assert any(name.startswith("Microsoft Word 365") for name in names)
    assert any(name.startswith("MS Excel") for name in names)
    assert any(name.startswith("MS Word") for name in names)
    assert "office_productivity" in expanded.matched_concepts
