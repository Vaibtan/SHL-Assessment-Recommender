# Purpose: Compare handler — grounded explainer over two catalog items.

from __future__ import annotations

import json

import structlog

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
    item_a = index.resolve_name(name_a, score_cutoff=FUZZY_MATCH_THRESHOLD)
    item_b = index.resolve_name(name_b, score_cutoff=FUZZY_MATCH_THRESHOLD)

    if item_a is None or item_b is None:
        fallbacks.append("compare_target_not_in_catalog")
        s_a = index.suggest_name(name_a, score_cutoff=70)
        s_b = index.suggest_name(name_b, score_cutoff=70)
        missing = []
        if item_a is None:
            missing.append(f"'{name_a}'" + (f" — did you mean '{s_a}'?" if s_a else ""))
        if item_b is None:
            missing.append(f"'{name_b}'" + (f" — did you mean '{s_b}'?" if s_b else ""))
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
