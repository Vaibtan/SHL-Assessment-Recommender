# Purpose: Refuse handler — canned templates per category, no LLM call.

from __future__ import annotations

from shl_recommender.agent.handlers._base import HandlerResult
from shl_recommender.agent.router import RefuseCategory, RouterDecision

_TEMPLATES: dict[RefuseCategory, str] = {
    RefuseCategory.INJECTION: (
        "I'm scoped to recommending SHL assessments. Tell me about the role "
        "you're hiring for and I'll help build a shortlist."
    ),
    RefuseCategory.OFF_TOPIC: (
        "I can help with SHL assessment selection — that's outside my scope. "
        "Want to share the role and seniority you're hiring for?"
    ),
    RefuseCategory.LEGAL: (
        "That's a legal question outside what I can advise on — your legal or "
        "compliance team is the right resource. I can help you select assessments; "
        "I just can't interpret regulatory obligations."
    ),
    RefuseCategory.GENERAL_ADVICE: (
        "I focus on assessment selection rather than broader hiring strategy. "
        "If you'd like, share the role and constraints and I'll shortlist relevant "
        "SHL assessments."
    ),
}

_DEFAULT_TEMPLATE = _TEMPLATES[RefuseCategory.OFF_TOPIC]


async def handle_refuse(*, decision: RouterDecision) -> HandlerResult:
    return HandlerResult(
        reply_text=_TEMPLATES.get(decision.refuse_category, _DEFAULT_TEMPLATE),
        entity_ids=[],
    )
