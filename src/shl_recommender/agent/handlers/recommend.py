"""Recommend handler — selection over a closed candidate pool.

Strategy:
1. Run retrieval for the router-emitted search_query + filters + coverage.
2. Hand the candidates (with entity_ids) to Gemini Flash.
3. Flash emits {"entity_ids": [...], "reply": "..."} via response_schema.
4. Validation drops any id not in candidates.

We deliberately keep this single-shot rather than tool-using — for recommend,
retrieval is one step and the LLM's job is selection. If selection fails to
return valid ids, we fall back to top retrieval candidates by RRF score.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field

from shl_recommender.agent.handlers._base import HandlerResult, messages_to_contents
from shl_recommender.agent.llm import LLMClient, LLMError, user_part
from shl_recommender.agent.prompts import RECOMMEND_SYSTEM_PROMPT
from shl_recommender.agent.router import RouterDecision
from shl_recommender.agent.tools import _full_summary
from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.catalog.normalize import CatalogItem
from shl_recommender.catalog.query_expansion import expand_catalog_candidates
from shl_recommender.catalog.retrieval import RetrievalFilters
from shl_recommender.config import get_settings
from shl_recommender.features.pipeline import FeatureBundle
from shl_recommender.schemas import Message

log = structlog.get_logger(__name__)

# Hard cap on candidate pool we send to the LLM — keeps prompt tight.
MAX_CANDIDATES_TO_LLM: int = 18

# Hard cap on emitted shortlist size from the model. API spec allows 10; we
# cap to 8 to keep room for refine-style additions in subsequent turns.
MAX_SHORTLIST: int = 8


class _SelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_ids: list[str] = Field(default_factory=list)
    reply: str = ""


async def handle_recommend(
    *,
    messages: list[Message],
    decision: RouterDecision,
    features: FeatureBundle,
    llm: LLMClient,
    index: CatalogIndex,
) -> HandlerResult:
    fallbacks: list[str] = []
    candidate_expansion = expand_catalog_candidates(
        _conversation_text(messages, decision),
        index.items,
        limit=MAX_SHORTLIST,
    )
    alias_candidates = list(candidate_expansion.items)
    candidates = await _build_candidate_pool_async(decision, features, index, llm)
    candidates = _merge_prioritized_candidates(alias_candidates, candidates)
    retrieval_stats = {
        "candidates_after_filter": len(candidates),
        "coverage_letters": list(decision.coverage_letters or _default_coverage(features)),
        "filters_relaxed": [],
        "expanded_candidates": len(alias_candidates),
        "matched_concepts": list(candidate_expansion.matched_concepts),
        "matched_aliases": list(candidate_expansion.matched_aliases[:12]),
    }

    if not candidates:
        # Try filter relaxation: drop duration_max first, then language, then test_type.
        candidates = await _relax_and_retry_async(decision, features, index, llm, fallbacks)
        candidates = _merge_prioritized_candidates(alias_candidates, candidates)
        retrieval_stats["candidates_after_relaxation"] = len(candidates)
        retrieval_stats["filters_relaxed"] = [
            f.split(":", 1)[1] for f in fallbacks if f.startswith("filters_relaxed:")
        ]

    if not candidates:
        fallbacks.append("no_candidates_after_relaxation")
        return HandlerResult(
            reply_text=(
                "I couldn't find catalog items matching those constraints. "
                "Could you relax one of: duration, language, or test type?"
            ),
            entity_ids=[],
            fallbacks_triggered=fallbacks,
            retrieval_stats=retrieval_stats,
        )

    candidate_summaries = _serialize_candidates(candidates, index)

    contents = messages_to_contents(messages)
    contents.append(user_part(_compose_user_payload(decision, candidate_summaries)))

    settings = get_settings()
    try:
        result = await llm.generate_structured(
            model=settings.handler_model,
            contents=contents,
            response_schema=_SelectionOutput,
            system_instruction=RECOMMEND_SYSTEM_PROMPT,
            temperature=settings.recommend_temperature,
            top_p=settings.top_p,
            max_output_tokens=2048,
        )
    except LLMError as e:
        log.warning("recommend_llm_failed_using_top_k", error=str(e))
        fallbacks.append("recommend_llm_fallback_to_top_k")
        return _top_k_fallback(candidates, index, fallbacks, retrieval_stats)

    if not result.text:
        fallbacks.append("recommend_empty_response_using_top_k")
        return _top_k_fallback(candidates, index, fallbacks, retrieval_stats)

    try:
        selection = _SelectionOutput.model_validate_json(result.text)
    except Exception as e:
        log.warning("recommend_invalid_json_using_top_k", error=str(e))
        fallbacks.append("recommend_invalid_json")
        salvaged = _salvage_selection_ids(result.text or "", candidates, fallbacks)
        if salvaged:
            return HandlerResult(
                reply_text=_default_reply(len(salvaged)),
                entity_ids=salvaged,
                fallbacks_triggered=fallbacks,
                retrieval_stats=retrieval_stats,
                validation_errors=["salvaged_ids_from_invalid_json"],
            )
        return _top_k_fallback(candidates, index, fallbacks, retrieval_stats)

    candidate_ids = {c.entity_id for c in candidates}
    valid: list[str] = []
    dropped_ids: list[str] = []
    for eid in selection.entity_ids:
        if eid in candidate_ids and eid not in valid:
            valid.append(eid)
        elif eid not in candidate_ids:
            dropped_ids.append(eid)
        if len(valid) >= MAX_SHORTLIST:
            break
    valid = _promote_alias_ids(valid, alias_candidates)

    if not valid:
        fallbacks.append("recommend_no_valid_ids_using_top_k")
        return _top_k_fallback(candidates, index, fallbacks, retrieval_stats)

    reply_text = selection.reply.strip() or _default_reply(len(valid))
    return HandlerResult(
        reply_text=reply_text,
        entity_ids=valid,
        fallbacks_triggered=fallbacks,
        retrieval_stats=retrieval_stats,
        validation_errors=[f"dropped_invalid_or_out_of_pool_ids:{len(dropped_ids)}"]
        if dropped_ids
        else [],
    )


# --------------------------------------------------------------------------------------
# Candidate pool construction
# --------------------------------------------------------------------------------------


async def _build_candidate_pool_async(
    decision: RouterDecision,
    features: FeatureBundle,
    index: CatalogIndex,
    llm: LLMClient,
    *,
    relaxation: tuple[str, ...] = (),
) -> list[CatalogItem]:
    """Async request-path candidate generation.

    Query embedding is the only network-bound step. It must not run synchronously
    on the FastAPI event loop; CPU retrieval is pushed to a worker thread.
    """
    filters = _to_filters(decision, relaxation)
    coverage = tuple(decision.coverage_letters) or _default_coverage(features)
    query = decision.search_query or features.latest_user_message
    query_vec = await _embed_query(llm, query)
    return await asyncio.to_thread(
        _retrieve_candidate_items,
        query,
        query_vec,
        filters,
        coverage,
        index,
    )


async def _embed_query(llm: LLMClient, query: str) -> np.ndarray | None:
    if not query or not query.strip():
        return None
    try:
        vectors = await llm.embed([query])
    except LLMError as e:
        log.warning("query_embed_failed", error=str(e))
        return None
    except Exception as e:  # pragma: no cover - defensive
        log.warning("query_embed_failed", error=repr(e))
        return None
    if not vectors:
        return None
    arr = np.asarray(vectors[0], dtype=np.float32)
    expected_dims = get_settings().embedding_dims
    if arr.shape != (expected_dims,):
        log.warning("query_embed_bad_shape", actual=arr.shape, expected=expected_dims)
        return None
    from shl_recommender.catalog.retrieval import l2_normalize

    return l2_normalize(arr.reshape(1, -1))[0]


def _retrieve_candidate_items(
    query: str,
    query_vec: np.ndarray | None,
    filters: RetrievalFilters,
    coverage: tuple[str, ...],
    index: CatalogIndex,
) -> list[CatalogItem]:
    hits = index.retriever.retrieve(
        query=query,
        query_vec=query_vec,
        filters=filters,
        coverage_letters=coverage,
        final_k=MAX_CANDIDATES_TO_LLM,
    )
    items = [index.get(h.entity_id) for h in hits]
    return [it for it in items if it is not None]


async def _relax_and_retry_async(
    decision: RouterDecision,
    features: FeatureBundle,
    index: CatalogIndex,
    llm: LLMClient,
    fallbacks: list[str],
) -> list[CatalogItem]:
    for relaxation in (
        ("duration_max",),
        ("duration_max", "languages"),
        ("duration_max", "languages", "test_types"),
        ("duration_max", "languages", "test_types", "job_levels"),
        (
            "duration_max",
            "languages",
            "test_types",
            "job_levels",
            "remote_only",
            "adaptive_only",
        ),
    ):
        items = await _build_candidate_pool_async(
            decision, features, index, llm, relaxation=relaxation
        )
        if items:
            fallbacks.append(f"filters_relaxed:{','.join(relaxation)}")
            return items
    return []


def _to_filters(decision: RouterDecision, relaxation: tuple[str, ...]) -> RetrievalFilters:
    f = decision.filters
    return RetrievalFilters(
        test_types=tuple(f.test_types) if "test_types" not in relaxation else (),
        languages=tuple(f.languages) if "languages" not in relaxation else (),
        job_levels=tuple(f.job_levels) if "job_levels" not in relaxation else (),
        duration_max_minutes=f.duration_max_minutes if "duration_max" not in relaxation else None,
        remote_only=f.remote_only if "remote_only" not in relaxation else False,
        adaptive_only=f.adaptive_only if "adaptive_only" not in relaxation else False,
    )


def _default_coverage(features: FeatureBundle) -> tuple[str, ...]:
    """When the router didn't specify coverage, default to P+A for fresh recommends."""
    if features.has_prior_shortlist:
        return ()
    return ("P", "A")


def _serialize_candidates(items: list[CatalogItem], index: CatalogIndex) -> list[dict[str, Any]]:
    return [_full_summary(it) for it in items]


def _conversation_text(messages: list[Message], decision: RouterDecision) -> str:
    user_text = "\n".join(m.content for m in messages if m.role == "user")
    return "\n".join(
        part
        for part in (
            user_text,
            decision.search_query,
            " ".join(decision.coverage_letters or ()),
            " ".join(decision.filters.test_types),
        )
        if part
    ).lower()


def _merge_prioritized_candidates(
    prioritized: list[CatalogItem],
    retrieved: list[CatalogItem],
) -> list[CatalogItem]:
    merged: list[CatalogItem] = []
    seen: set[str] = set()
    for item in (*prioritized, *retrieved):
        if item.entity_id in seen:
            continue
        merged.append(item)
        seen.add(item.entity_id)
        if len(merged) >= MAX_CANDIDATES_TO_LLM:
            break
    return merged


def _promote_alias_ids(selected: list[str], alias_candidates: list[CatalogItem]) -> list[str]:
    promoted: list[str] = []
    for item in alias_candidates:
        if item.entity_id not in promoted:
            promoted.append(item.entity_id)
        if len(promoted) >= MAX_SHORTLIST:
            return promoted
    for eid in selected:
        if eid not in promoted:
            promoted.append(eid)
        if len(promoted) >= MAX_SHORTLIST:
            break
    return promoted


def _compose_user_payload(decision: RouterDecision, candidates: list[dict[str, Any]]) -> str:
    return (
        "ROUTER_DECISION:\n"
        f"{decision.model_dump_json(indent=2)}\n\n"
        "CANDIDATES (closed set — pick entity_ids only from this list):\n"
        f"{json.dumps(candidates, indent=2, ensure_ascii=False)}\n\n"
        "Emit JSON: {\"entity_ids\": [...], \"reply\": \"...\"}"
    )


# --------------------------------------------------------------------------------------
# Fallback paths
# --------------------------------------------------------------------------------------


def _top_k_fallback(
    candidates: list[CatalogItem],
    index: CatalogIndex,
    fallbacks: list[str],
    retrieval_stats: dict[str, Any] | None = None,
) -> HandlerResult:
    ids = [it.entity_id for it in candidates[:5]]
    return HandlerResult(
        reply_text="Here are the closest catalog matches based on retrieval.",
        entity_ids=ids,
        fallbacks_triggered=fallbacks,
        retrieval_stats=retrieval_stats or {"candidates_after_filter": len(candidates)},
    )


def _default_reply(n: int) -> str:
    return f"Here are {n} assessments that fit your needs."


def _salvage_selection_ids(
    raw_text: str,
    candidates: list[CatalogItem],
    fallbacks: list[str],
) -> list[str]:
    """Recover closed-set entity IDs from a truncated JSON prefix.

    This is not a primary parse path; it only runs after Pydantic rejects the
    structured response. The closed candidate pool still enforces catalog-only
    IDs and ordering from the model output.
    """
    candidate_ids = {it.entity_id for it in candidates}
    valid: list[str] = []
    for eid in re.findall(r'"(\d+)"', raw_text):
        if eid in candidate_ids and eid not in valid:
            valid.append(eid)
        if len(valid) >= MAX_SHORTLIST:
            break
    if valid:
        fallbacks.append("salvaged_ids_from_invalid_json")
    return valid
