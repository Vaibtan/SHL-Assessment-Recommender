"""Unit tests for catalog normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shl_recommender.catalog.normalize import (
    KEY_TO_LETTER,
    build_search_text,
    normalize_catalog,
    normalize_record,
)

CATALOG_JSON = Path(__file__).resolve().parents[2] / "data" / "shl_product_catalog.json"


def test_full_catalog_normalizes() -> None:
    items = normalize_catalog(CATALOG_JSON)
    assert len(items) >= 350
    for it in items:
        assert it.entity_id
        assert it.name
        assert it.url.startswith("https://www.shl.com/")
        assert it.test_type
        for letter in it.test_type.split(","):
            assert letter in KEY_TO_LETTER.values()


def test_test_type_letters_match_keys_order() -> None:
    rec = {
        "entity_id": "x",
        "name": "Demo",
        "link": "https://www.shl.com/x",
        "keys": ["Knowledge & Skills", "Simulations"],
        "description": "irrelevant",
    }
    item = normalize_record(rec)
    assert item is not None
    assert item.test_type == "K,S"


def test_test_type_drops_unknown_keys() -> None:
    rec = {
        "entity_id": "x",
        "name": "Demo",
        "link": "https://www.shl.com/x",
        "keys": ["Knowledge & Skills", "Bogus Category"],
    }
    item = normalize_record(rec)
    assert item is not None
    assert item.test_type == "K"


def test_record_with_no_keys_is_dropped() -> None:
    rec = {
        "entity_id": "x",
        "name": "Demo",
        "link": "https://www.shl.com/x",
        "keys": [],
    }
    assert normalize_record(rec) is None


def test_record_missing_required_fields_is_dropped() -> None:
    assert normalize_record({}) is None
    assert normalize_record({"entity_id": "x"}) is None
    assert normalize_record({"entity_id": "x", "name": "y"}) is None  # no link


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("13 minutes", 13),
        ("Approximate Completion Time in minutes = 30", 30),
        ("", None),
        ("Untimed", None),
        ("Variable", None),
        ("—", None),
    ],
)
def test_duration_parsing(raw: str, expected: int | None) -> None:
    rec = {
        "entity_id": "x",
        "name": "Demo",
        "link": "https://www.shl.com/x",
        "keys": ["Knowledge & Skills"],
        "duration": raw,
    }
    item = normalize_record(rec)
    assert item is not None
    assert item.duration_minutes == expected


def test_control_chars_are_stripped_from_description() -> None:
    rec = {
        "entity_id": "x",
        "name": "Demo",
        "link": "https://www.shl.com/x",
        "keys": ["Knowledge & Skills"],
        "description": "First line.\x07\x01 Second line.\x1f With control bytes.",
    }
    item = normalize_record(rec)
    assert item is not None
    assert "\x07" not in item.description
    assert "\x01" not in item.description
    assert "\x1f" not in item.description
    assert "First line" in item.description and "Second line" in item.description


def test_search_text_contains_labeled_fields() -> None:
    rec = {
        "entity_id": "x",
        "name": "Java 8 (New)",
        "link": "https://www.shl.com/x",
        "keys": ["Knowledge & Skills"],
        "job_levels": ["Mid-Professional"],
        "languages": ["English (USA)"],
        "duration": "30 minutes",
        "description": "tests Java 8 features",
    }
    item = normalize_record(rec)
    assert item is not None
    text = build_search_text(item)
    assert "NAME: Java 8 (New)" in text
    assert "KEYS: Knowledge & Skills" in text
    assert "JOB_LEVELS: Mid-Professional" in text
    assert "LANGUAGES: English (USA)" in text
    assert "DURATION: 30 minutes" in text


def test_catalog_dedupes_by_entity_id(tmp_path: Path) -> None:
    payload = [
        {
            "entity_id": "1",
            "name": "A",
            "link": "https://www.shl.com/a",
            "keys": ["Knowledge & Skills"],
        },
        {
            "entity_id": "1",  # duplicate
            "name": "A2",
            "link": "https://www.shl.com/a2",
            "keys": ["Knowledge & Skills"],
        },
        {
            "entity_id": "2",
            "name": "B",
            "link": "https://www.shl.com/b",
            "keys": ["Personality & Behavior"],
        },
    ]
    p = tmp_path / "cat.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    items = normalize_catalog(p)
    assert [i.entity_id for i in items] == ["1", "2"]
