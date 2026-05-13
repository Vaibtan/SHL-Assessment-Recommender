# Purpose: Clarify handler — one Flash call, no tools, returns a single question.

from __future__ import annotations

import structlog

from shl_recommender.agent.handlers._base import HandlerResult, messages_to_contents
from shl_recommender.agent.llm import LLMClient, LLMError
from shl_recommender.agent.prompts import CLARIFY_SYSTEM_PROMPT
from shl_recommender.agent.router import RouterDecision
from shl_recommender.config import get_settings
from shl_recommender.features.pipeline import FeatureBundle
from shl_recommender.schemas import Message

log = structlog.get_logger(__name__)

_DEFAULT_CLARIFY = "What role are you hiring for, and at what seniority?"


async def handle_clarify(
    *,
    messages: list[Message],
    decision: RouterDecision,
    features: FeatureBundle,
    llm: LLMClient,
) -> HandlerResult:
    """Generate one clarifying question.

    Trust the router's clarifying_question if it produced one; otherwise call
    Flash to write one. No tools.
    """
    fallbacks: list[str] = []

    if decision.clarifying_question.strip():
        return HandlerResult(reply_text=decision.clarifying_question.strip(), fallbacks_triggered=fallbacks)

    contents = messages_to_contents(messages)
    settings = get_settings()
    try:
        result = await llm.generate_text(
            model=settings.handler_model,
            contents=contents,
            system_instruction=CLARIFY_SYSTEM_PROMPT,
            temperature=settings.clarify_temperature,
            top_p=settings.top_p,
            max_output_tokens=128,
        )
    except LLMError as e:
        log.warning("clarify_llm_failed_using_default", error=str(e))
        fallbacks.append("clarify_default_used")
        return HandlerResult(reply_text=_DEFAULT_CLARIFY, fallbacks_triggered=fallbacks)

    text = (result.text or "").strip()
    if not text:
        fallbacks.append("clarify_empty_response")
        text = _DEFAULT_CLARIFY
    text = text.split("\n")[0].strip().strip('"')
    if len(text) > 240:
        text = text[:237].rstrip() + "…"
    return HandlerResult(reply_text=text, fallbacks_triggered=fallbacks)
