# Purpose: Shared handler abstractions — types every handler returns + a tool-loop runner.

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import structlog
from google.genai import types as gtypes

from shl_recommender.agent.llm import (
    LLMCallResult,
    LLMClient,
    LLMError,
    function_call_to_dict,
    function_response_part,
    user_part,
)
from shl_recommender.agent.tools import ToolBox, dispatch
from shl_recommender.config import get_settings
from shl_recommender.schemas import Message

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class HandlerResult:
    """Uniform handler return — assembly layer materializes ids → recommendations."""

    reply_text: str
    entity_ids: list[str] = field(default_factory=list)
    fallbacks_triggered: list[str] = field(default_factory=list)
    tool_calls_made: int = 0
    retrieval_stats: dict[str, Any] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)


def messages_to_contents(messages: Sequence[Message]) -> list[gtypes.Content]:
    """Map API messages -> Gemini Content objects (roles 'user'/'model')."""
    out: list[gtypes.Content] = []
    for m in messages:
        if m.role == "user":
            out.append(gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=m.content)]))
        elif m.role == "assistant":
            out.append(gtypes.Content(role="model", parts=[gtypes.Part.from_text(text=m.content)]))
    return out


async def run_tool_loop(
    *,
    llm: LLMClient,
    toolbox: ToolBox,
    contents: list[gtypes.Content],
    tools: list[gtypes.Tool],
    system_instruction: str,
    max_iterations: int = 3,
    temperature: float = 0.1,
    fallback_factory: Callable[[Exception], LLMCallResult] | None = None,
) -> tuple[LLMCallResult, list[str], int]:
    """Run a tool-using inner agent loop.

    Returns: (final_result, fallbacks_triggered, tool_calls_made).
    The caller is responsible for parsing `final_result.text` into the
    handler-specific output schema.
    """
    fallbacks: list[str] = []
    tool_calls = 0
    cur_contents = list(contents)

    for step in range(max_iterations):
        try:
            result = await llm.generate_with_tools(
                model=get_settings().handler_model,
                contents=cur_contents,
                tools=tools,
                system_instruction=system_instruction,
                temperature=temperature,
            )
        except LLMError as e:
            log.warning("handler_llm_failed", step=step, error=str(e))
            fallbacks.append("llm_call_failed")
            if fallback_factory is not None:
                return fallback_factory(e), fallbacks, tool_calls
            raise

        if not result.function_calls:
            return result, fallbacks, tool_calls

        tool_calls += len(result.function_calls)

        tool_responses = await asyncio.gather(
            *(_execute_tool(toolbox, fc) for fc in result.function_calls)
        )

        model_parts = [
            gtypes.Part(function_call=fc) for fc in result.function_calls
        ]
        cur_contents.append(gtypes.Content(role="model", parts=model_parts))
        for fc, resp in zip(result.function_calls, tool_responses):
            cur_contents.append(function_response_part(fc.name, resp))

    fallbacks.append("max_iterations_reached")
    log.info("handler_max_iterations", iters=max_iterations)
    try:
        result = await llm.generate_text(
            model=get_settings().handler_model,
            contents=cur_contents + [user_part("Provide your final answer now as JSON.")],
            system_instruction=system_instruction,
            temperature=temperature,
        )
    except LLMError as e:
        if fallback_factory is not None:
            return fallback_factory(e), fallbacks + ["fallback_factory_used"], tool_calls
        raise
    return result, fallbacks, tool_calls


async def _execute_tool(toolbox: ToolBox, fc: gtypes.FunctionCall) -> dict[str, Any]:
    args = dict(fc.args or {})
    log.debug("tool_call", **function_call_to_dict(fc))
    return await asyncio.to_thread(dispatch, toolbox, fc.name, args)
