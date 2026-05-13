# Purpose: Test fakes — a deterministic stand-in for LLMClient.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

import numpy as np
from google.genai import types as gtypes

from shl_recommender.agent.llm import LLMCallResult, LLMClient
from shl_recommender.config import get_settings


@dataclass
class StubReply:
    """A scripted reply for one LLM call."""

    text: str | None = None
    function_calls: list[gtypes.FunctionCall] = field(default_factory=list)
    finish_reason: str = "STOP"


def stub_text(text: str) -> StubReply:
    return StubReply(text=text)


def stub_json(payload: dict[str, Any]) -> StubReply:
    return StubReply(text=json.dumps(payload))


def stub_function_call(name: str, args: dict[str, Any]) -> StubReply:
    return StubReply(
        text=None,
        function_calls=[gtypes.FunctionCall(name=name, args=args)],
    )


class FakeLLMClient(LLMClient):
    """Returns scripted replies in order for each call site (router/handler)."""

    def __init__(self) -> None:
        self.router_replies: list[StubReply] = []
        self.handler_replies: list[StubReply] = []
        self.embed_dim: int = get_settings().embedding_dims
        self.call_log: list[dict[str, Any]] = []

    def _get_client(self):  # type: ignore[override]
        raise RuntimeError("FakeLLMClient does not call the network")

    async def generate_structured(self, **kwargs):  # type: ignore[override]
        self.call_log.append({"site": "structured", **{k: v for k, v in kwargs.items() if k != "contents"}})
        replies = self.router_replies if _is_router(kwargs) else self.handler_replies
        return _pop(replies)

    async def generate_with_tools(self, **kwargs):  # type: ignore[override]
        self.call_log.append({"site": "tools", **{k: v for k, v in kwargs.items() if k != "contents"}})
        return _pop(self.handler_replies)

    async def generate_text(self, **kwargs):  # type: ignore[override]
        self.call_log.append({"site": "text", **{k: v for k, v in kwargs.items() if k != "contents"}})
        return _pop(self.handler_replies)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:  # type: ignore[override]
        return self.embed_sync(texts)

    def embed_sync(self, texts: Sequence[str]) -> list[list[float]]:  # type: ignore[override]
        out: list[list[float]] = []
        for t in texts:
            seed = sum(ord(c) for c in t) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.embed_dim).astype(np.float32)
            v = v / max(np.linalg.norm(v), 1e-9)
            out.append(v.tolist())
        return out


def _is_router(kwargs: dict) -> bool:
    """Heuristic — the router prompt is the only one with 'policy layer' in it."""
    sys = kwargs.get("system_instruction") or ""
    return "policy layer" in sys.lower()


def _pop(replies: list[StubReply]) -> LLMCallResult:
    if not replies:
        return LLMCallResult(text=None, function_calls=[], finish_reason="STOP")
    r = replies.pop(0)
    return LLMCallResult(
        text=r.text,
        function_calls=list(r.function_calls),
        finish_reason=r.finish_reason,
    )
