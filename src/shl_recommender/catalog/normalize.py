# Purpose: Catalog normalization — JSON → clean records → parquet.

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

KEY_TO_LETTER: Final[dict[str, str]] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

LETTER_TO_KEY: Final[dict[str, str]] = {v: k for k, v in KEY_TO_LETTER.items()}

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True, slots=True)
class CatalogItem:
    """Normalized assessment record — the canonical in-memory shape."""

    entity_id: str
    name: str
    url: str
    description: str
    keys: tuple[str, ...]
    test_type: str  # comma-joined single-letter codes (e.g. "K,S")
    job_levels: tuple[str, ...]
    languages: tuple[str, ...]
    duration: str  # e.g. "13 minutes" or "" if unknown
    duration_minutes: int | None  # parsed numeric duration, None if unknown/variable
    remote: bool
    adaptive: bool


_DURATION_NUMERIC_RE = re.compile(r"(\d+)")


def _sanitize(s: str) -> str:
    """Strip control chars and normalize whitespace; preserve unicode."""
    if not s:
        return ""
    cleaned = _CONTROL_CHAR_RE.sub(" ", s)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    return " ".join(cleaned.split()).strip()


def _parse_duration(raw: str) -> int | None:
    """Extract an integer-minutes value from a duration string.

    Returns None for empty / "Variable" / "Untimed" / unparseable values.
    """
    if not raw:
        return None
    if any(token in raw.lower() for token in ("variable", "untimed", "—")):
        return None
    match = _DURATION_NUMERIC_RE.search(raw)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _to_test_type_code(keys: Iterable[str]) -> str:
    """Map ordered `keys` to a comma-joined single-letter code.

    Unknown categories are dropped silently. Empty input yields an empty
    string — caller decides how to handle.
    """
    letters: list[str] = []
    seen: set[str] = set()
    for k in keys:
        letter = KEY_TO_LETTER.get(k)
        if letter and letter not in seen:
            letters.append(letter)
            seen.add(letter)
    return ",".join(letters)


def _yes_no(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if not value:
        return False
    return str(value).strip().lower() in {"yes", "true", "1"}


def parse_raw_catalog(json_path: Path) -> list[dict]:
    """Load the raw JSON tolerating embedded control characters."""
    raw = json_path.read_text(encoding="utf-8")
    return json.loads(raw, strict=False)


def normalize_record(rec: dict) -> CatalogItem | None:
    """Convert a raw record into a CatalogItem, or return None if invalid.

    Items missing entity_id, name, link, or having empty `keys` are dropped —
    they cannot satisfy the API contract (`test_type` would be empty).
    """
    entity_id = str(rec.get("entity_id", "")).strip()
    name = _sanitize(str(rec.get("name", "")))
    url = str(rec.get("link", "")).strip()
    if not entity_id or not name or not url:
        return None

    raw_keys = rec.get("keys", []) or []
    keys = tuple(str(k) for k in raw_keys if isinstance(k, str))
    test_type = _to_test_type_code(keys)
    if not test_type:
        return None

    description = _sanitize(str(rec.get("description", "")))
    job_levels = tuple(str(j) for j in (rec.get("job_levels") or []) if isinstance(j, str))
    languages = tuple(str(l) for l in (rec.get("languages") or []) if isinstance(l, str))
    duration_raw = str(rec.get("duration", "") or "").strip()
    duration_minutes = _parse_duration(duration_raw)

    return CatalogItem(
        entity_id=entity_id,
        name=name,
        url=url,
        description=description,
        keys=keys,
        test_type=test_type,
        job_levels=job_levels,
        languages=languages,
        duration=duration_raw,
        duration_minutes=duration_minutes,
        remote=_yes_no(rec.get("remote")),
        adaptive=_yes_no(rec.get("adaptive")),
    )


def normalize_catalog(json_path: Path) -> list[CatalogItem]:
    """End-to-end: load raw JSON → list of clean CatalogItem records."""
    raw_records = parse_raw_catalog(json_path)
    items: list[CatalogItem] = []
    seen_ids: set[str] = set()
    for rec in raw_records:
        item = normalize_record(rec)
        if item is None or item.entity_id in seen_ids:
            continue
        items.append(item)
        seen_ids.add(item.entity_id)
    return items


def build_search_text(item: CatalogItem, description_chars: int = 1200) -> str:
    """Compose the text indexed for BM25 / dense retrieval.

    Field labels are kept so BM25 can latch onto them, and the description is
    truncated so a long item doesn't drown shorter ones in TF.
    """
    description = item.description[:description_chars] if item.description else ""
    parts = [
        f"NAME: {item.name}",
        f"KEYS: {', '.join(item.keys)}" if item.keys else "",
        f"JOB_LEVELS: {', '.join(item.job_levels)}" if item.job_levels else "",
        f"LANGUAGES: {', '.join(item.languages)}" if item.languages else "",
        f"DURATION: {item.duration}" if item.duration else "",
        f"DESCRIPTION: {description}" if description else "",
    ]
    return "\n".join(p for p in parts if p)


def to_parquet_records(items: Sequence[CatalogItem]) -> list[dict]:
    """Serialize CatalogItems to dict-rows suitable for parquet."""
    return [
        {
            "entity_id": it.entity_id,
            "name": it.name,
            "url": it.url,
            "description": it.description,
            "keys": list(it.keys),
            "test_type": it.test_type,
            "job_levels": list(it.job_levels),
            "languages": list(it.languages),
            "duration": it.duration,
            "duration_minutes": it.duration_minutes,
            "remote": it.remote,
            "adaptive": it.adaptive,
        }
        for it in items
    ]


def from_parquet_records(records: list[dict]) -> list[CatalogItem]:
    """Inverse of `to_parquet_records`. Handles numpy arrays / NaN from pandas."""
    items: list[CatalogItem] = []
    for r in records:
        items.append(
            CatalogItem(
                entity_id=str(r["entity_id"]),
                name=str(r["name"]),
                url=str(r["url"]),
                description=_str_or_empty(r.get("description")),
                keys=_to_str_tuple(r.get("keys")),
                test_type=str(r["test_type"]),
                job_levels=_to_str_tuple(r.get("job_levels")),
                languages=_to_str_tuple(r.get("languages")),
                duration=_str_or_empty(r.get("duration")),
                duration_minutes=_int_or_none(r.get("duration_minutes")),
                remote=bool(r.get("remote", False)),
                adaptive=bool(r.get("adaptive", False)),
            )
        )
    return items


def _str_or_empty(v: object) -> str:
    """Coerce a possibly-None / NaN value to string."""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN check without importing numpy
        return ""
    return str(v)


def _to_str_tuple(v: object) -> tuple[str, ...]:
    """Coerce any iterable (list / numpy array / None) to a tuple of strings."""
    if v is None:
        return ()
    if isinstance(v, str):
        return (v,)
    try:
        return tuple(str(x) for x in v)  # type: ignore[arg-type]
    except TypeError:
        return ()


def _int_or_none(v: object) -> int | None:
    """Coerce a value to int, mapping NaN/None/empty to None."""
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
