# Purpose: Gemini client wrapper — Vertex AI auth, async, retry, structured output, tools.

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

import structlog
from google import genai
from google.genai import types as gtypes
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from shl_recommender.config import get_settings

log = structlog.get_logger(__name__)

_LLM_STATS: ContextVar[dict[str, Any] | None] = ContextVar("llm_stats", default=None)


def __getattr__(name: str) -> object:
    settings = get_settings()
    mapping = {
        "ROUTER_MODEL": settings.router_model,
        "HANDLER_MODEL": settings.handler_model,
        "EMBEDDING_MODEL": settings.embedding_model,
        "EMBEDDING_DIMS": settings.embedding_dims,
    }
    if name in mapping:
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class LLMError(RuntimeError):
    """Raised when the underlying SDK fails after retries."""


def begin_llm_stats() -> Token:
    """Start collecting per-request LLM stats in the current context."""
    return _LLM_STATS.set(
        {"count": 0, "tokens_in": 0, "tokens_out": 0, "models": {}, "timeouts": 0}
    )


def end_llm_stats(token: Token) -> dict[str, Any]:
    """Return collected LLM stats and restore the previous context."""
    stats = _LLM_STATS.get() or {}
    _LLM_STATS.reset(token)
    models = stats.get("models", {})
    return {
        "count": int(stats.get("count", 0)),
        "tokens_in": int(stats.get("tokens_in", 0)),
        "tokens_out": int(stats.get("tokens_out", 0)),
        "models": sorted(models),
        "timeouts": int(stats.get("timeouts", 0)),
    }



_UNSUPPORTED_KEYS: frozenset[str] = frozenset(
    {"additionalProperties", "additional_properties", "$defs", "definitions", "title", "default"}
)


def _normalize_schema(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_normalize_schema(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref_name = node["$ref"].rsplit("/", 1)[-1]
        target = defs.get(ref_name)
        if target is not None:
            resolved = _normalize_schema(target, defs)
            sibling = {k: v for k, v in node.items() if k != "$ref"}
            if sibling and isinstance(resolved, dict):
                resolved = {**resolved, **_normalize_schema(sibling, defs)}
            return resolved

    if "anyOf" in node and isinstance(node["anyOf"], list):
        variants = node["anyOf"]
        non_null = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
        if len(non_null) == 1 and len(non_null) != len(variants):
            base = _normalize_schema(non_null[0], defs)
            if isinstance(base, dict):
                base = {**base, "nullable": True}
                for k, v in node.items():
                    if k == "anyOf" or k in _UNSUPPORTED_KEYS:
                        continue
                    if k not in base:
                        base[k] = _normalize_schema(v, defs)
                return base

    cleaned: dict[str, Any] = {}
    for k, v in node.items():
        if k in _UNSUPPORTED_KEYS:
            continue
        cleaned[k] = _normalize_schema(v, defs)
    return cleaned


def pydantic_to_gemini_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to a JSON-Schema dict Gemini accepts.

    Drops `additionalProperties`/`title`/`default`, inlines `$ref`/`$defs`, and
    rewrites optional fields from `anyOf` to `nullable: true`.
    """
    raw = model_cls.model_json_schema()
    defs = raw.get("$defs") or raw.get("definitions") or {}
    return _normalize_schema(raw, defs)


def _is_transient(exc: BaseException) -> bool:
    """Identify transient errors worth retrying.

    Conservative: anything that looks like a network hiccup, rate limit, or 5xx.
    """
    if isinstance(exc, TimeoutError):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "timeout",
            "deadline",
            "unavailable",
            "rate limit",
            "resource exhausted",
            "503",
            "504",
            "502",
            "connection",
        )
    )


@dataclass(slots=True)
class LLMCallResult:
    """Normalized result from a Gemini call.

    Either `text` is set (text/structured-output response) or `function_calls` is
    non-empty (tool-call response). Token counts are best-effort from usage metadata.
    """

    text: str | None
    function_calls: list[gtypes.FunctionCall] = field(default_factory=list)
    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    raw: Any = None  # the GenerateContentResponse, for debugging


class LLMClient:
    """Async-first Gemini client honoring the locked architecture.

    Auth precedence: Vertex AI (GOOGLE_CLOUD_PROJECT env) → AI Studio (API key).
    """

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        self._api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GCP_API_KEY")
        )
        self._client: genai.Client | None = None
        self._client_lock = threading.Lock()

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            if self._project and self._api_key:
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project,
                    location=self._location,
                    api_key=self._api_key,
                )
            elif self._project:
                self._client = genai.Client(
                    vertexai=True, project=self._project, location=self._location
                )
            elif self._api_key:
                self._client = genai.Client(api_key=self._api_key)
            else:
                raise LLMError(
                    "No Gemini credentials. Set GOOGLE_CLOUD_PROJECT (+ ADC or API key for Vertex), "
                    "or GOOGLE_API_KEY / GEMINI_API_KEY / GCP_API_KEY (AI Studio)."
                )
            return self._client

    async def generate_structured(
        self,
        *,
        model: str,
        contents: list[gtypes.Content] | str,
        response_schema: dict[str, Any] | type,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_output_tokens: int = 1024,
    ) -> LLMCallResult:
        """One-shot structured-output call. No tool calling.

        `response_schema` can be a Pydantic BaseModel class or a pre-shaped dict.
        Pydantic classes are converted via `pydantic_to_gemini_schema` so AI Studio
        (which rejects `additionalProperties`, raw `$ref`, and `anyOf`-with-null)
        accepts the payload alongside Vertex AI.
        """
        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            schema_arg: dict[str, Any] | type = pydantic_to_gemini_schema(response_schema)
        else:
            schema_arg = response_schema

        config = gtypes.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_arg,
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            thinking_config=_thinking_disabled(),
            automatic_function_calling=_automatic_function_calling_disabled(),
        )
        return await self._call(
            model=model,
            contents=contents,
            config=config,
            schema_compliance_check=_json_schema_compliant,
        )

    async def generate_with_tools(
        self,
        *,
        model: str,
        contents: list[gtypes.Content],
        tools: list[gtypes.Tool],
        system_instruction: str | None = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_output_tokens: int = 2048,
        tool_mode: str = "AUTO",
    ) -> LLMCallResult:
        """Tool-using call. Returns either text or function_calls (mutually exclusive)."""
        tool_config = gtypes.ToolConfig(
            function_calling_config=gtypes.FunctionCallingConfig(
                mode=tool_mode,  # AUTO | ANY | NONE
            )
        )
        config = gtypes.GenerateContentConfig(
            tools=tools,
            tool_config=tool_config,
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            thinking_config=_thinking_disabled(),
            automatic_function_calling=_automatic_function_calling_disabled(),
        )
        return await self._call(model=model, contents=contents, config=config)

    async def generate_text(
        self,
        *,
        model: str,
        contents: list[gtypes.Content] | str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_output_tokens: int = 1024,
    ) -> LLMCallResult:
        """Plain text generation — no schema, no tools."""
        config = gtypes.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            thinking_config=_thinking_disabled(),
            automatic_function_calling=_automatic_function_calling_disabled(),
        )
        return await self._call(model=model, contents=contents, config=config)

    def embed_sync(self, texts: Sequence[str]) -> list[list[float]]:
        """Synchronous embed — for sync code paths (e.g. retrieval inside async handlers).

        Calling `embed()` from inside the running event loop via
        `run_coroutine_threadsafe` deadlocks the loop, so sync callers must use
        this method directly.
        """
        client = self._get_client()
        settings = get_settings()
        config = gtypes.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=settings.embedding_dims,
        )
        result = client.models.embed_content(
            model=settings.embedding_model, contents=list(texts), config=config
        )
        return [list(emb.values) for emb in result.embeddings]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts via the configured embedding model + dims."""
        loop = asyncio.get_running_loop()
        settings = get_settings()
        started = time.perf_counter()
        try:
            vectors = await asyncio.wait_for(
                loop.run_in_executor(None, self.embed_sync, list(texts)),
                timeout=settings.llm_timeout_seconds,
            )
        except TimeoutError:
            _record_timeout()
            raise LLMError(
                f"Gemini embedding timed out after {settings.llm_timeout_seconds}s"
            ) from None
        _record_llm_call(
            model=settings.embedding_model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason="EMBED",
            tool_calls_count=0,
            tokens_in=None,
            tokens_out=None,
        )
        return vectors

    async def _call(
        self,
        *,
        model: str,
        contents: list[gtypes.Content] | str,
        config: gtypes.GenerateContentConfig,
        schema_compliance_check: Callable[[LLMCallResult], bool] | None = None,
    ) -> LLMCallResult:
        client = self._get_client()
        loop = asyncio.get_running_loop()
        settings = get_settings()

        async def _invoke() -> Any:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model, contents=contents, config=config
                    ),
                ),
                timeout=settings.llm_timeout_seconds,
            )

        attempts = 0
        started = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.3, max=2.0),
                retry=retry_if_exception(_is_transient),
                reraise=True,
            ):
                with attempt:
                    attempts = attempt.retry_state.attempt_number
                    raw = await _invoke()
        except RetryError as e:
            raise LLMError(f"Gemini call failed after retries: {e}") from e
        except TimeoutError as e:
            _record_timeout()
            raise LLMError(
                f"Gemini call timed out after {settings.llm_timeout_seconds}s"
            ) from e
        except Exception as e:
            raise LLMError(f"Gemini call error: {e}") from e

        normalized = _normalize_response(raw)
        schema_compliant = (
            schema_compliance_check(normalized)
            if schema_compliance_check is not None
            else None
        )
        _record_llm_call(
            model=model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=normalized.finish_reason,
            tool_calls_count=len(normalized.function_calls),
            tokens_in=normalized.tokens_in,
            tokens_out=normalized.tokens_out,
            retry_count=max(0, attempts - 1),
            schema_compliant=schema_compliant,
        )
        return normalized


def _record_timeout() -> None:
    stats = _LLM_STATS.get()
    if stats is not None:
        stats["timeouts"] = int(stats.get("timeouts", 0)) + 1


def _record_llm_call(
    *,
    model: str,
    latency_ms: int,
    finish_reason: str | None,
    tool_calls_count: int,
    tokens_in: int | None,
    tokens_out: int | None,
    retry_count: int = 0,
    schema_compliant: bool | None = None,
) -> None:
    stats = _LLM_STATS.get()
    if stats is not None:
        stats["count"] = int(stats.get("count", 0)) + 1
        stats["tokens_in"] = int(stats.get("tokens_in", 0)) + int(tokens_in or 0)
        stats["tokens_out"] = int(stats.get("tokens_out", 0)) + int(tokens_out or 0)
        models = stats.setdefault("models", {})
        models[model] = int(models.get(model, 0)) + 1

    log.debug(
        "llm_call",
        model=model,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        tool_calls_count=tool_calls_count,
        retry_count=retry_count,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        schema_compliant=schema_compliant,
    )


def _json_schema_compliant(result: LLMCallResult) -> bool:
    if not result.text:
        return False
    try:
        json.loads(result.text)
    except json.JSONDecodeError:
        return False
    return True


def _thinking_disabled() -> gtypes.ThinkingConfig:
    """Disable Gemini 2.5 thinking so JSON budgets are spent on visible output."""
    return gtypes.ThinkingConfig(thinking_budget=0)


def _automatic_function_calling_disabled() -> gtypes.AutomaticFunctionCallingConfig:
    """Avoid SDK automatic-function-calling side effects for explicit agent loops."""
    return gtypes.AutomaticFunctionCallingConfig(disable=True)


def _normalize_response(raw: Any) -> LLMCallResult:
    """Best-effort extraction across SDK versions."""
    text: str | None = None
    function_calls: list[gtypes.FunctionCall] = []
    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None

    try:
        if raw.candidates:
            cand = raw.candidates[0]
            finish_reason = getattr(cand, "finish_reason", None)
            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc and (getattr(fc, "name", None) or getattr(fc, "args", None)):
                        function_calls.append(fc)
                    text_part = getattr(part, "text", None)
                    if text_part:
                        text = (text or "") + text_part
        usage = getattr(raw, "usage_metadata", None)
        if usage is not None:
            tokens_in = getattr(usage, "prompt_token_count", None)
            tokens_out = getattr(usage, "candidates_token_count", None)
    except Exception:
        pass

    return LLMCallResult(
        text=text,
        function_calls=function_calls,
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        raw=raw,
    )




def user_part(text: str) -> gtypes.Content:
    """Build a Content object representing a user turn."""
    return gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=text)])


def model_part(text: str) -> gtypes.Content:
    """Build a Content object representing a model turn."""
    return gtypes.Content(role="model", parts=[gtypes.Part.from_text(text=text)])


def function_response_part(name: str, response: dict[str, Any]) -> gtypes.Content:
    """Build a Content object carrying a tool-call result back to the model."""
    return gtypes.Content(
        role="user",
        parts=[
            gtypes.Part.from_function_response(name=name, response=response)
        ],
    )


def function_call_to_dict(fc: gtypes.FunctionCall) -> dict[str, Any]:
    """Stable serializer for a function-call (logs / tests)."""
    return {"name": fc.name, "args": dict(fc.args or {})}
