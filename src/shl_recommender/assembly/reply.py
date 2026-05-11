"""Reply assembly — markdown shortlist embedding + end_of_conversation invariant."""

from __future__ import annotations

from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.catalog.normalize import CatalogItem
from shl_recommender.schemas import ChatResponse, Recommendation


def compute_end_of_conversation(is_final_turn_flag: bool, recommendations: list[Recommendation]) -> bool:
    """Apply the policy invariant: end=true only when a shortlist exists."""
    return bool(is_final_turn_flag) and len(recommendations) > 0


def render_shortlist_table(items: list[CatalogItem]) -> str:
    """Render the shortlist as a markdown table mirroring sample_conversations format."""
    if not items:
        return ""
    rows = [
        "| # | Name | Test Type | Keys | Duration | Languages | URL |",
        "|---|------|-----------|------|----------|-----------|-----|",
    ]
    for i, it in enumerate(items, start=1):
        rows.append(
            f"| {i} | {it.name} | {it.test_type} | {_languages_summary(it.keys, max_n=5)} | "
            f"{it.duration or '—'} | {_languages_summary(it.languages, max_n=4)} | <{it.url}> |"
        )
    return "\n".join(rows)


def _languages_summary(values: tuple[str, ...] | list[str], max_n: int) -> str:
    if not values:
        return "—"
    head = list(values)[:max_n]
    extra = len(values) - len(head)
    suffix = f" _(+{extra} more)_" if extra > 0 else ""
    return ", ".join(head) + suffix


def assemble_chat_response(
    *,
    reply_text: str,
    entity_ids: list[str],
    is_final_turn_flag: bool,
    index: CatalogIndex,
) -> ChatResponse:
    """Final assembly — build ChatResponse with embedded markdown table."""
    from shl_recommender.assembly.validator import materialize  # local to keep deps light

    recommendations = materialize(entity_ids, index)
    items = [it for it in (index.get(r.url and _id_for(index, r)) for r in recommendations) if it is not None]
    # ↑ a clean re-resolve; keeps Recommendation list and table aligned by entity_id order.

    body = reply_text.rstrip()
    if recommendations:
        table = render_shortlist_table(items if items else _items_for_recs(recommendations, index))
        if table:
            body = f"{body}\n\n{table}" if body else table

    end_of_conv = compute_end_of_conversation(is_final_turn_flag, recommendations)
    return ChatResponse(
        reply=body or "—",
        recommendations=recommendations,
        end_of_conversation=end_of_conv,
    )


def _id_for(index: CatalogIndex, rec: Recommendation) -> str:
    """Resolve a Recommendation back to its entity_id by URL match (safer than name)."""
    for it in index.items:
        if it.url == rec.url:
            return it.entity_id
    return ""


def _items_for_recs(recs: list[Recommendation], index: CatalogIndex) -> list[CatalogItem]:
    out: list[CatalogItem] = []
    for r in recs:
        for it in index.items:
            if it.url == r.url:
                out.append(it)
                break
    return out
