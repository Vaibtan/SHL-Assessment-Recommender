# Purpose: Validator — enforce closed-set IDs and materialize API recommendations.

from __future__ import annotations

from typing import Iterable

from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.schemas import Recommendation

MAX_RECOMMENDATIONS: int = 10


def validate_ids(ids: Iterable[str], index: CatalogIndex) -> list[str]:
    """Return only ids that resolve to a real catalog item, preserving order; deduped."""
    valid: list[str] = []
    seen: set[str] = set()
    for eid in ids:
        if eid in seen:
            continue
        if index.get(eid) is not None:
            valid.append(eid)
            seen.add(eid)
    return valid


def materialize(ids: Iterable[str], index: CatalogIndex) -> list[Recommendation]:
    """Convert validated ids into API-shaped Recommendation objects.

    Truncates at MAX_RECOMMENDATIONS to satisfy the spec.
    """
    out: list[Recommendation] = []
    for eid in ids:
        item = index.get(eid)
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
