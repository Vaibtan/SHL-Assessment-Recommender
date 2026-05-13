# Purpose: Router — Policy layer.

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import structlog
from pydantic import BaseModel, ConfigDict, Field

from shl_recommender.agent.llm import LLMClient, LLMError, model_part, user_part
from shl_recommender.agent.prompts import ROUTER_SYSTEM_PROMPT
from shl_recommender.config import get_settings
from shl_recommender.features.pipeline import FeatureBundle
from shl_recommender.schemas import Message

log = structlog.get_logger(__name__)


class Intent(str, Enum):
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    REFUSE = "refuse"


class RefuseCategory(str, Enum):
    INJECTION = "injection"
    OFF_TOPIC = "off_topic"
    LEGAL = "legal"
    GENERAL_ADVICE = "general_advice"


class RouterFilters(BaseModel):
    """Filter facets the router may emit."""

    model_config = ConfigDict(extra="forbid")

    test_types: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    job_levels: list[str] = Field(default_factory=list)
    duration_max_minutes: int | None = None
    remote_only: bool = False
    adaptive_only: bool = False


class ComparePair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: str
    b: str


class ConstraintSwap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    from_: str = Field(alias="from")
    to: str


class ConstraintDeltas(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    add: list[str] = Field(default_factory=list)
    drop: list[str] = Field(default_factory=list)
    swap: list[ConstraintSwap] = Field(default_factory=list)


class RouterDecision(BaseModel):
    """Structured output the router emits."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    search_query: str = ""
    filters: RouterFilters = Field(default_factory=RouterFilters)
    coverage_letters: list[str] = Field(default_factory=list)
    compare_pair: ComparePair | None = None
    clarifying_question: str = ""
    refuse_category: RefuseCategory | None = None
    refuse_reason: str = ""
    constraint_deltas: ConstraintDeltas = Field(default_factory=ConstraintDeltas)
    is_final_turn: bool = False




def heuristic_decision(features: FeatureBundle) -> RouterDecision:
    """Conservative deterministic fallback if the LLM call fails outright."""
    if features.injection_signal:
        return RouterDecision(
            intent=Intent.REFUSE,
            refuse_category=RefuseCategory.INJECTION,
            refuse_reason="Detected an attempt to override scope.",
        )
    if features.off_topic_signal:
        return RouterDecision(
            intent=Intent.REFUSE,
            refuse_category=RefuseCategory.OFF_TOPIC,
            refuse_reason="That's outside SHL assessment selection.",
        )
    if features.legal_signal:
        return RouterDecision(
            intent=Intent.REFUSE,
            refuse_category=RefuseCategory.LEGAL,
            refuse_reason="That's a legal question outside what I can advise on.",
        )
    if (
        features.vagueness_score >= 0.6
        and not features.has_prior_shortlist
        and features.turns_remaining > 2
    ):
        return RouterDecision(
            intent=Intent.CLARIFY,
            clarifying_question="What role are you hiring for, and at what seniority?",
        )
    if features.has_prior_shortlist and features.last_user_confirmation:
        return RouterDecision(
            intent=Intent.REFINE,
            search_query=features.latest_user_message,
            is_final_turn=True,
        )
    return RouterDecision(
        intent=Intent.RECOMMEND,
        search_query=features.latest_user_message or "",
        coverage_letters=["P", "A"] if not features.has_prior_shortlist else [],
    )




@dataclass(frozen=True, slots=True)
class RouteRequest:
    """Bundle of inputs the router needs."""

    messages: Sequence[Message]
    features: FeatureBundle


def _format_messages_for_prompt(messages: Sequence[Message]) -> list:
    """Adapt API messages to Gemini Content objects (role mapping)."""
    out = []
    for m in messages:
        if m.role == "user":
            out.append(user_part(m.content))
        elif m.role == "assistant":
            out.append(model_part(m.content))
    return out


def _format_user_payload(features: FeatureBundle) -> str:
    """The router gets the conversation as turns + a JSON sidecar with features."""
    return (
        "FEATURE_BUNDLE:\n"
        f"{json.dumps(features.summary_for_prompt(), indent=2)}\n\n"
        "Emit your routing decision as JSON conforming to the schema."
    )


async def route(req: RouteRequest, llm: LLMClient) -> RouterDecision:
    """Run the policy layer. Returns a typed RouterDecision."""
    contents = _format_messages_for_prompt(req.messages)
    contents.append(user_part(_format_user_payload(req.features)))

    settings = get_settings()
    try:
        result = await llm.generate_structured(
            model=settings.router_model,
            contents=contents,
            response_schema=RouterDecision,
            system_instruction=ROUTER_SYSTEM_PROMPT,
            temperature=settings.router_temperature,
            top_p=settings.top_p,
            max_output_tokens=2048,
        )
    except LLMError as e:
        log.warning("router_llm_failed_using_heuristic", error=str(e))
        return heuristic_decision(req.features)

    if not result.text:
        log.warning("router_empty_response_using_heuristic")
        return heuristic_decision(req.features)

    try:
        decision = RouterDecision.model_validate_json(result.text)
    except Exception as e:
        log.warning("router_invalid_json_using_heuristic", error=str(e))
        return heuristic_decision(req.features)

    if (
        decision.is_final_turn
        and not req.features.has_prior_shortlist
        and req.features.last_user_confirmation
        and req.features.vagueness_score >= 0.6
    ):
        return RouterDecision(
            intent=Intent.CLARIFY,
            clarifying_question="What role are you hiring for, and at what seniority?",
            is_final_turn=False,
        )

    if decision.is_final_turn and not (
        req.features.has_prior_shortlist
        or decision.intent in (Intent.RECOMMEND, Intent.REFINE)
    ):
        decision = decision.model_copy(update={"is_final_turn": False})
    return decision
