"""Validator — enforce closed-set IDs and materialize API recommendations.

This module never trusts IDs from outside the catalog. Any ID not present in
the index is dropped silently (logged at the call site).
"""

from __future__ import annotations

from typing import Iterable

from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.catalog.normalize import CatalogItem
from shl_recommender.schemas import Recommendation

# Hard cap from the API spec.
MAX_RECOMMENDATIONS: int = 10


def validate_ids(ids: Iterable[str], index: CatalogIndex) -> list[str]:
    """Return only ids that resolve to a real catalog item, preserving order; deduped."""
    valid: list[str] = []
    seen: set[str] = set()
    by_id = {it.entity_id: it for it in index.items}
    for eid in ids:
        if eid in seen:
            continue
        if eid in by_id:
            valid.append(eid)
            seen.add(eid)
    return valid


def materialize(ids: Iterable[str], index: CatalogIndex) -> list[Recommendation]:
    """Convert validated ids into API-shaped Recommendation objects.

    Truncates at MAX_RECOMMENDATIONS to satisfy the spec.
    """
    by_id: dict[str, CatalogItem] = {it.entity_id: it for it in index.items}
    out: list[Recommendation] = []
    for eid in ids:
        item = by_id.get(eid)
        if item is None:
            continue
        out.append(
            Recommendation(
                name=item.name,
                url=item.url,
                test_type=item.test_type,
            )
        )
        if len(out) >= MAX_RECOMMENDATIONS:
            break
    return out
