"""Compare handler — grounded explainer over two catalog items."""

from __future__ import annotations

import asyncio
import json

import structlog
from rapidfuzz import fuzz, process

from shl_recommender.agent.handlers._base import HandlerResult, messages_to_contents
from shl_recommender.agent.llm import LLMClient, LLMError, user_part
from shl_recommender.agent.prompts import COMPARE_SYSTEM_PROMPT
from shl_recommender.agent.router import RouterDecision
from shl_recommender.agent.tools import _full_summary
from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.catalog.normalize import CatalogItem
from shl_recommender.config import get_settings
from shl_recommender.features.pipeline import FeatureBundle
from shl_recommender.schemas import Message

log = structlog.get_logger(__name__)

# rapidfuzz score threshold for accepting a fuzzy name match.
FUZZY_MATCH_THRESHOLD: int = 85


async def handle_compare(
    *,
    messages: list[Message],
    decision: RouterDecision,
    features: FeatureBundle,
    llm: LLMClient,
    index: CatalogIndex,
) -> HandlerResult:
    fallbacks: list[str] = []
    if decision.compare_pair is None:
        fallbacks.append("compare_no_pair")
        return HandlerResult(
            reply_text="Which two assessments would you like to compare?",
            entity_ids=list(features.prior_shortlist_ids),
            fallbacks_triggered=fallbacks,
        )

    name_a, name_b = decision.compare_pair.a, decision.compare_pair.b
    item_a, item_b = await asyncio.gather(
        asyncio.to_thread(_resolve, name_a, index.items),
        asyncio.to_thread(_resolve, name_b, index.items),
    )

    if item_a is None or item_b is None:
        fallbacks.append("compare_target_not_in_catalog")
        suggestions = await asyncio.gather(
            asyncio.to_thread(_suggest, name_a, index.items),
            asyncio.to_thread(_suggest, name_b, index.items),
        )
        s_a, s_b = suggestions
        missing = []
        if item_a is None:
            missing.append(f"'{name_a}'" + (f" — did you mean {s_a}?" if s_a else ""))
        if item_b is None:
            missing.append(f"'{name_b}'" + (f" — did you mean {s_b}?" if s_b else ""))
        reply = (
            "I couldn't find " + " and ".join(missing) + " in the SHL catalog."
            if missing
            else "Couldn't resolve the comparison targets."
        )
        return HandlerResult(
            reply_text=reply,
            entity_ids=list(features.prior_shortlist_ids),
            fallbacks_triggered=fallbacks,
        )

    contents = messages_to_contents(messages)
    contents.append(user_part(_compose_payload(item_a, item_b)))

    settings = get_settings()
    try:
        result = await llm.generate_text(
            model=settings.handler_model,
            contents=contents,
            system_instruction=COMPARE_SYSTEM_PROMPT,
            temperature=settings.compare_temperature,
            top_p=settings.top_p,
            max_output_tokens=512,
        )
    except LLMError as e:
        log.warning("compare_llm_failed", error=str(e))
        fallbacks.append("compare_llm_fallback")
        return HandlerResult(
            reply_text=_static_compare(item_a, item_b),
            entity_ids=list(features.prior_shortlist_ids),
            fallbacks_triggered=fallbacks,
        )

    text = (result.text or "").strip()
    if not text:
        fallbacks.append("compare_empty_response")
        text = _static_compare(item_a, item_b)

    return HandlerResult(
        reply_text=text,
        entity_ids=list(features.prior_shortlist_ids),
        fallbacks_triggered=fallbacks,
    )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _resolve(name: str, items: list[CatalogItem]) -> CatalogItem | None:
    if not name:
        return None
    target = name.strip().lower()
    for it in items:
        if it.name.lower() == target:
            return it
    choices = {it.entity_id: it.name for it in items}
    match = process.extractOne(name, choices, scorer=fuzz.WRatio, score_cutoff=FUZZY_MATCH_THRESHOLD)
    if match is None:
        return None
    _, _, eid = match
    return next((it for it in items if it.entity_id == eid), None)


def _suggest(name: str, items: list[CatalogItem]) -> str | None:
    """Return the closest catalog name to `name`, or None if nothing close."""
    if not name:
        return None
    choices = {it.entity_id: it.name for it in items}
    match = process.extractOne(name, choices, scorer=fuzz.WRatio, score_cutoff=70)
    if match is None:
        return None
    matched_name, _, _ = match
    return f"'{matched_name}'"


def _compose_payload(a: CatalogItem, b: CatalogItem) -> str:
    return (
        "ITEM_A:\n"
        f"{json.dumps(_full_summary(a), indent=2, ensure_ascii=False)}\n\n"
        "ITEM_B:\n"
        f"{json.dumps(_full_summary(b), indent=2, ensure_ascii=False)}\n\n"
        "Write the grounded comparison."
    )


def _static_compare(a: CatalogItem, b: CatalogItem) -> str:
    """Last-resort comparison built from the records when the LLM fails."""
    return (
        f"{a.name} ({a.test_type}) vs {b.name} ({b.test_type}). "
        f"{a.name}: {a.description[:240]}… "
        f"{b.name}: {b.description[:240]}…"
    )
