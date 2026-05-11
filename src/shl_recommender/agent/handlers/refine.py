"""Refine handler — modify an existing shortlist based on user deltas."""

from __future__ import annotations

import json
import re
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from shl_recommender.agent.handlers._base import HandlerResult, messages_to_contents
from shl_recommender.agent.handlers.recommend import (
    MAX_CANDIDATES_TO_LLM,
    MAX_SHORTLIST,
    _build_candidate_pool_async,
    _serialize_candidates,
)
from shl_recommender.agent.llm import LLMClient, LLMError, user_part
from shl_recommender.agent.prompts import REFINE_SYSTEM_PROMPT
from shl_recommender.agent.router import RouterDecision
from shl_recommender.agent.tools import _full_summary
from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.config import get_settings
from shl_recommender.features.pipeline import FeatureBundle
from shl_recommender.schemas import Message

log = structlog.get_logger(__name__)


class _RefineOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_ids: list[str] = Field(default_factory=list)
    reply: str = ""


async def handle_refine(
    *,
    messages: list[Message],
    decision: RouterDecision,
    features: FeatureBundle,
    llm: LLMClient,
    index: CatalogIndex,
) -> HandlerResult:
    fallbacks: list[str] = []
    prior_ids = list(features.prior_shortlist_ids)
    prior_items = [index.get(eid) for eid in prior_ids]
    prior_items = [it for it in prior_items if it is not None]

    # If user merely confirmed (empty deltas), preserve the prior shortlist as-is.
    if (
        not decision.constraint_deltas.add
        and not decision.constraint_deltas.drop
        and not decision.constraint_deltas.swap
    ):
        if prior_items:
            return HandlerResult(
                reply_text="Confirmed.",
                entity_ids=[it.entity_id for it in prior_items],
                fallbacks_triggered=fallbacks,
            )
        # No prior + no deltas means we should not be in refine — fall back to recommend.
        fallbacks.append("refine_no_prior_falling_back_to_recommend")
        from shl_recommender.agent.handlers.recommend import handle_recommend  # local to avoid cycle

        return await handle_recommend(
            messages=messages,
            decision=decision,
            features=features,
            llm=llm,
            index=index,
        )

    # Re-retrieve with merged constraints (router has already composed search_query).
    new_candidates = await _build_candidate_pool_async(decision, features, index, llm)
    retrieval_stats = {
        "prior_count": len(prior_items),
        "new_candidates": len(new_candidates),
        "pool_size": 0,
    }

    # Build a combined candidate pool: prior shortlist (preserved) ∪ new candidates.
    pool: list[Any] = []
    seen: set[str] = set()
    for it in prior_items:
        if it.entity_id not in seen:
            pool.append(it)
            seen.add(it.entity_id)
    for it in new_candidates:
        if it.entity_id not in seen:
            pool.append(it)
            seen.add(it.entity_id)
        if len(pool) >= MAX_CANDIDATES_TO_LLM + len(prior_items):
            break
    retrieval_stats["pool_size"] = len(pool)

    candidate_summaries = _serialize_candidates(pool, index)
    prior_summaries = [_full_summary(it) for it in prior_items]

    contents = messages_to_contents(messages)
    contents.append(user_part(_compose_payload(decision, prior_summaries, candidate_summaries)))

    settings = get_settings()
    try:
        result = await llm.generate_structured(
            model=settings.handler_model,
            contents=contents,
            response_schema=_RefineOutput,
            system_instruction=REFINE_SYSTEM_PROMPT,
            temperature=settings.refine_temperature,
            top_p=settings.top_p,
            max_output_tokens=2048,
        )
    except LLMError as e:
        log.warning("refine_llm_failed_falling_back", error=str(e))
        fallbacks.append("refine_llm_fallback_to_prior")
        return _prior_fallback(prior_items, fallbacks, retrieval_stats)

    if not result.text:
        fallbacks.append("refine_empty_response_using_prior")
        return _prior_fallback(prior_items, fallbacks, retrieval_stats)

    try:
        out = _RefineOutput.model_validate_json(result.text)
    except Exception as e:
        log.warning("refine_invalid_json_using_prior", error=str(e))
        fallbacks.append("refine_invalid_json")
        salvaged = _salvage_refine_ids(result.text or "", pool, fallbacks)
        if salvaged:
            return HandlerResult(
                reply_text="Updated.",
                entity_ids=salvaged,
                fallbacks_triggered=fallbacks,
                retrieval_stats=retrieval_stats,
                validation_errors=["salvaged_ids_from_invalid_json"],
            )
        return _prior_fallback(prior_items, fallbacks, retrieval_stats)

    pool_ids = {it.entity_id for it in pool}
    valid: list[str] = []
    dropped_ids: list[str] = []
    for eid in out.entity_ids:
        if eid in pool_ids and eid not in valid:
            valid.append(eid)
        elif eid not in pool_ids:
            dropped_ids.append(eid)
        if len(valid) >= MAX_SHORTLIST:
            break

    if not valid:
        fallbacks.append("refine_no_valid_ids_using_prior")
        return _prior_fallback(prior_items, fallbacks, retrieval_stats)

    reply_text = out.reply.strip() or "Updated."
    return HandlerResult(
        reply_text=reply_text,
        entity_ids=valid,
        fallbacks_triggered=fallbacks,
        retrieval_stats=retrieval_stats,
        validation_errors=[f"dropped_invalid_or_out_of_pool_ids:{len(dropped_ids)}"]
        if dropped_ids
        else [],
    )


def _compose_payload(
    decision: RouterDecision,
    prior: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    return (
        "ROUTER_DECISION:\n"
        f"{decision.model_dump_json(indent=2)}\n\n"
        "PRIOR_SHORTLIST (must be preserved unless deltas modify):\n"
        f"{json.dumps(prior, indent=2, ensure_ascii=False)}\n\n"
        "CANDIDATES (additional items to consider for adds/swaps):\n"
        f"{json.dumps(candidates, indent=2, ensure_ascii=False)}\n\n"
        "Emit JSON: {\"entity_ids\": [...], \"reply\": \"...\"}"
    )


def _prior_fallback(
    prior_items: list[Any],
    fallbacks: list[str],
    retrieval_stats: dict[str, Any] | None = None,
) -> HandlerResult:
    if not prior_items:
        return HandlerResult(
            reply_text="I had trouble updating the shortlist. Could you describe the role again?",
            entity_ids=[],
            fallbacks_triggered=fallbacks,
            retrieval_stats=retrieval_stats or {},
        )
    return HandlerResult(
        reply_text="Keeping the prior shortlist.",
        entity_ids=[it.entity_id for it in prior_items],
        fallbacks_triggered=fallbacks,
        retrieval_stats=retrieval_stats or {},
    )


def _salvage_refine_ids(raw_text: str, pool: list[Any], fallbacks: list[str]) -> list[str]:
    pool_ids = {it.entity_id for it in pool}
    valid: list[str] = []
    for eid in re.findall(r'"(\d+)"', raw_text):
        if eid in pool_ids and eid not in valid:
            valid.append(eid)
        if len(valid) >= MAX_SHORTLIST:
            break
    if valid:
        fallbacks.append("salvaged_ids_from_invalid_json")
    return valid
