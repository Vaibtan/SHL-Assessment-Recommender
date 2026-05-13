# Purpose: Pre-router feature pipeline.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final, Sequence

import numpy as np

from shl_recommender.agent.llm import LLMClient
from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.catalog.retrieval import RetrievalHit, l2_normalize
from shl_recommender.schemas import Message

MAX_TURNS: Final[int] = 8

_CONFIRMATION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bperfect\b",
        r"\bconfirmed\b",
        r"\bconfirm\b",
        r"\blocked in\b",
        r"\blocking it in\b",
        r"\bthat(?:'s| is) good\b",
        r"\bthat works\b",
        r"\bsounds good\b",
        r"\bfinal list\b",
        r"\bfinal battery\b",
        r"\bkeep it\b",
        r"\bkeep the shortlist\b",
        r"\bthanks?\b",
        r"\bship it\b",
        r"\bgo with that\b",
        r"\bthat(?:'s| is) what we need\b",
        r"\bthat covers it\b",
        r"\bwe(?:'ll| will) (?:use|go with)\b",
        r"\bgood (?:two-stage|choice)\b",
    )
)

_INJECTION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (?:all )?(?:previous|prior) instructions?",
        r"disregard (?:all )?(?:previous|prior) instructions?",
        r"system prompt",
        r"reveal (?:the |your )?prompt",
        r"forget (?:everything|your instructions)",
        r"you are now",
        r"act as (?:a |an )?(?:hacker|admin|developer|jailbreak)",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"developer mode",
    )
)

_OFF_TOPIC_KEYWORDS: Final[tuple[str, ...]] = (
    "weather",
    "stock price",
    "recipe",
    "movie",
    "joke",
    "song",
    "lyrics",
    "homework",
    "essay",
    "translate this",
    "write code for",
    "play a game",
    "cricket",
    "football",
)

_LEGAL_KEYWORDS: Final[tuple[str, ...]] = (
    r"legally required",
    r"legally obligated",
    r"satisf(?:y|ies)\s+(?:the|that|this|a|any)\s+\w*\s*requirement",
    r"\beeoc\b",
    r"\blawsuit\b",
    r"comply with the law",
    r"regulatory obligation",
    r"fulfil(?:s|l)?\s+\w*\s*regulatory",
)

_MD_LINK_RE: Final[re.Pattern[str]] = re.compile(
    r"<?(https?://www\.shl\.com/products/product-catalog/view/[^>\s|)]+)/?>?"
)


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    """Everything the router needs in addition to the conversation history."""

    turn_index: int  # number of messages so far (incl. latest user)
    turns_remaining: int  # 8 - turn_index (clamped >= 0)
    latest_user_message: str
    has_prior_shortlist: bool
    prior_shortlist_ids: tuple[str, ...]
    last_user_confirmation: bool
    vagueness_score: float  # 0.0 (specific) -> 1.0 (vague)
    injection_signal: bool
    off_topic_signal: bool
    legal_signal: bool
    peek_retrieval: tuple[RetrievalHit, ...]  # top-3 from BM25-only on latest msg

    def summary_for_prompt(self) -> dict:
        """Compact serialization for inlining in the router prompt."""
        return {
            "turn_index": self.turn_index,
            "turns_remaining": self.turns_remaining,
            "has_prior_shortlist": self.has_prior_shortlist,
            "prior_shortlist_ids": list(self.prior_shortlist_ids),
            "last_user_confirmation": self.last_user_confirmation,
            "vagueness_score": round(self.vagueness_score, 2),
            "injection_signal": self.injection_signal,
            "off_topic_signal": self.off_topic_signal,
            "legal_signal": self.legal_signal,
            "peek_retrieval": [
                {"entity_id": h.entity_id, "score": round(h.score, 4)}
                for h in self.peek_retrieval
            ],
        }




def latest_user_message(messages: Sequence[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def latest_assistant_message(messages: Sequence[Message]) -> str:
    for m in reversed(messages):
        if m.role == "assistant":
            return m.content
    return ""


def parse_prior_shortlist(messages: Sequence[Message], index: CatalogIndex) -> list[str]:
    """Recover entity_ids from the most recent assistant markdown table.

    The agent embeds a markdown table of the active shortlist in every commit
    turn. We walk back through assistant turns and pull URLs (more reliable than
    names — URL is the catalog primary key by content).

    Returns ordered, deduplicated entity_ids.
    """
    for m in reversed(messages):
        if m.role != "assistant":
            continue
        urls = _MD_LINK_RE.findall(m.content)
        if urls:
            ids = _resolve_urls(urls, index)
            if ids:
                return ids
        ids = _parse_names_from_table(m.content, index)
        if ids:
            return ids
    return []


def is_confirmation(text: str) -> bool:
    """Detect user-confirmation tokens in the latest user message."""
    if not text:
        return False
    for pat in _CONFIRMATION_PATTERNS:
        if pat.search(text):
            return True
    return False


def vagueness_score(text: str) -> float:
    """Cheap heuristic: short messages with few content words are vague.

    0.0 = highly specific (>= 25 content tokens, contains role/skill keyword).
    1.0 = highly vague (<= 4 content tokens, generic phrasing).
    """
    if not text:
        return 1.0
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-+#.]*", text.lower())
    n = len(tokens)
    if n <= 4:
        return 1.0
    if n >= 25:
        return 0.0
    return max(0.0, min(1.0, (25 - n) / 21))


def injection_signal(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def off_topic_signal(text: str) -> bool:
    if not text:
        return False
    lc = text.lower()
    return any(kw in lc for kw in _OFF_TOPIC_KEYWORDS)


def legal_signal(text: str) -> bool:
    if not text:
        return False
    lc = text.lower()
    return any(re.search(kw, lc) for kw in _LEGAL_KEYWORDS)


def turn_budget(messages: Sequence[Message]) -> tuple[int, int]:
    """(turn_index, turns_remaining) — turn_index counts ALL messages so far."""
    n = len(messages)
    return n, max(0, MAX_TURNS - n)


def peek_retrieval(query: str, index: CatalogIndex, k: int = 3) -> list[RetrievalHit]:
    """BM25-only top-K — used as cheap router context (no embedding API call)."""
    if not query.strip():
        return []
    retriever = index.retriever
    return retriever.retrieve(query=query, query_vec=None, per_retriever_k=k, final_k=k)




async def build_feature_bundle(
    messages: Sequence[Message],
    index: CatalogIndex,
    llm: LLMClient | None = None,  # accepted for parity; embeddings are deferred to handlers
) -> FeatureBundle:
    """Compute deterministic router features.

    The function stays async because it is part of the agent pipeline contract,
    but the current features are deliberately cheap and run inline. Avoiding
    `asyncio.to_thread` here prevents threadpool churn under high concurrency.
    """
    user_msg = latest_user_message(messages)
    turn_index, turns_remaining = turn_budget(messages)
    prior_ids = parse_prior_shortlist(messages, index)
    confirmation = is_confirmation(user_msg)
    vague = vagueness_score(user_msg)
    injection = injection_signal(user_msg)
    off_topic = off_topic_signal(user_msg)
    legal = legal_signal(user_msg)
    peek_hits = peek_retrieval(user_msg, index, 3)

    return FeatureBundle(
        turn_index=turn_index,
        turns_remaining=turns_remaining,
        latest_user_message=user_msg,
        has_prior_shortlist=bool(prior_ids),
        prior_shortlist_ids=tuple(prior_ids),
        last_user_confirmation=confirmation,
        vagueness_score=vague,
        injection_signal=injection,
        off_topic_signal=off_topic,
        legal_signal=legal,
        peek_retrieval=tuple(peek_hits),
    )




def _resolve_urls(urls: list[str], index: CatalogIndex) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for u in urls:
        item = index.get_by_url(u)
        eid = item.entity_id if item is not None else None
        if eid and eid not in seen:
            out.append(eid)
            seen.add(eid)
    return out


def _parse_names_from_table(markdown: str, index: CatalogIndex) -> list[str]:
    """Fuzzy-match names from each table row's second column."""
    out: list[str] = []
    seen: set[str] = set()
    for line in markdown.splitlines():
        if not line.strip().startswith("|"):
            continue
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(parts) < 2:
            continue
        name = parts[1]
        if not name or name.startswith("---") or name.lower() == "name":
            continue
        name = re.sub(r"[*_`]", "", name).strip()
        item = index.resolve_name(name, score_cutoff=90)
        eid = item.entity_id if item is not None else None
        if eid and eid not in seen:
            out.append(eid)
            seen.add(eid)
    return out
