"""Replay harness — Recall@10 measurement loop.

Two modes:
- LIVE: uses a real LLMClient against Vertex / AI Studio (set GOOGLE_CLOUD_PROJECT
  or GEMINI_API_KEY in env). Slow (~60s per trace) and costs API tokens.
- SCRIPTED: uses a deterministic FakeLLMClient with pre-baked replies per
  sample. Used in CI / unit-style smoke runs to validate the pipeline shape.

The harness drives each persona against the agent for up to MAX_TURNS.
By default it uses the next prepared user-turn from the trace for deterministic
tests. In live mode, pass `user_llm` to simulate a dynamic user from the
persona facts, which is closer to the graded evaluator.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import structlog

from shl_recommender.agent.llm import LLMClient
from shl_recommender.config import get_settings
from shl_recommender.agent.runner import Agent
from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.features.pipeline import MAX_TURNS
from shl_recommender.schemas import ChatRequest, Message
from tests.replay.personas import SamplePersona, load_all_personas

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class TurnRecord:
    user: str
    assistant_reply: str
    intent: str
    is_final_turn: bool
    recommendations_count: int
    end_of_conversation: bool
    timings_ms: dict[str, int]


@dataclass(slots=True)
class TraceRecord:
    sample_id: str
    expected_ids: list[str]
    final_predicted_ids: list[str]
    recall_at_10: float
    schema_valid: bool
    end_reached: bool
    turns: list[TurnRecord] = field(default_factory=list)


@dataclass(slots=True)
class ReplayReport:
    traces: list[TraceRecord]
    mean_recall_at_10: float
    schema_pass_rate: float

    def to_dict(self) -> dict:
        return {
            "mean_recall_at_10": self.mean_recall_at_10,
            "schema_pass_rate": self.schema_pass_rate,
            "traces": [
                {
                    "sample_id": t.sample_id,
                    "expected_ids": t.expected_ids,
                    "final_predicted_ids": t.final_predicted_ids,
                    "recall_at_10": t.recall_at_10,
                    "schema_valid": t.schema_valid,
                    "end_reached": t.end_reached,
                    "turn_count": len(t.turns),
                }
                for t in self.traces
            ],
        }


async def replay_persona(
    persona: SamplePersona,
    agent: Agent,
    *,
    max_turns: int = MAX_TURNS,
    user_llm: LLMClient | None = None,
) -> TraceRecord:
    """Drive one persona through the agent up to `max_turns`."""
    messages: list[Message] = []
    next_user_idx = 0

    record = TraceRecord(
        sample_id=persona.sample_id,
        expected_ids=list(persona.expected_entity_ids),
        final_predicted_ids=[],
        recall_at_10=0.0,
        schema_valid=True,
        end_reached=False,
    )

    while len(messages) < max_turns:
        user_msg, next_user_idx = await _next_user_message(
            persona=persona,
            messages=messages,
            next_user_idx=next_user_idx,
            user_llm=user_llm,
        )

        messages.append(Message(role="user", content=user_msg))
        if len(messages) > max_turns:
            break

        t0 = time.perf_counter()
        try:
            agent_result = await agent.chat(ChatRequest(messages=messages))
        except Exception as e:
            log.exception("replay_agent_error", sample_id=persona.sample_id, error=str(e))
            record.schema_valid = False
            return record
        latency_ms = int((time.perf_counter() - t0) * 1000)

        response = agent_result.response
        record.turns.append(
            TurnRecord(
                user=user_msg,
                assistant_reply=response.reply,
                intent=agent_result.decision.intent.value,
                is_final_turn=agent_result.decision.is_final_turn,
                recommendations_count=len(response.recommendations),
                end_of_conversation=response.end_of_conversation,
                timings_ms={"total_ms": latency_ms, **agent_result.timings},
            )
        )
        messages.append(Message(role="assistant", content=response.reply))

        # Persist the latest non-empty shortlist as our final prediction.
        if response.recommendations:
            urls = [r.url for r in response.recommendations]
            record.final_predicted_ids = _ids_for_urls(urls, agent.index)

        if response.end_of_conversation:
            record.end_reached = True
            break

    record.recall_at_10 = _recall_at_k(record.final_predicted_ids, record.expected_ids, k=10)
    return record


def _ids_for_urls(urls: list[str], index: CatalogIndex) -> list[str]:
    by_url = {it.url.rstrip("/"): it.entity_id for it in index.items}
    out: list[str] = []
    for u in urls:
        eid = by_url.get(u.rstrip("/"))
        if eid is not None and eid not in out:
            out.append(eid)
    return out


def _recall_at_k(predicted: list[str], expected: list[str], k: int = 10) -> float:
    if not expected:
        return 1.0  # nothing to recall — degenerate case
    top_k = set(predicted[:k])
    hits = top_k & set(expected)
    return len(hits) / len(expected)


async def replay_all(
    agent: Agent,
    *,
    max_turns: int = MAX_TURNS,
    user_llm: LLMClient | None = None,
) -> ReplayReport:
    personas = load_all_personas(agent.index)
    records = []
    for p in personas:
        log.info("replay_start", sample_id=p.sample_id, expected=len(p.expected_entity_ids))
        rec = await replay_persona(p, agent, max_turns=max_turns, user_llm=user_llm)
        records.append(rec)
        log.info(
            "replay_done",
            sample_id=p.sample_id,
            recall_at_10=round(rec.recall_at_10, 3),
            schema_valid=rec.schema_valid,
            end_reached=rec.end_reached,
            turns=len(rec.turns),
        )

    mean_recall = sum(r.recall_at_10 for r in records) / max(1, len(records))
    schema_pass = sum(1 for r in records if r.schema_valid) / max(1, len(records))
    return ReplayReport(
        traces=records,
        mean_recall_at_10=round(mean_recall, 3),
        schema_pass_rate=round(schema_pass, 3),
    )


async def _next_user_message(
    *,
    persona: SamplePersona,
    messages: list[Message],
    next_user_idx: int,
    user_llm: LLMClient | None,
) -> tuple[str, int]:
    """Return the next simulated user turn.

    The first turn is always the trace opening. Deterministic tests continue to
    consume sample turns. Live replay uses an LLM user after turn one so the
    agent must handle realistic clarification and confirmation wording.
    """
    if not messages:
        if persona.user_turns:
            return persona.user_turns[0], 1
        return persona.opening_message, next_user_idx

    if user_llm is None:
        if next_user_idx >= len(persona.user_turns):
            return "Locking it in. Thanks.", next_user_idx
        return persona.user_turns[next_user_idx], next_user_idx + 1

    prompt = _simulated_user_prompt(persona, messages)
    settings = get_settings()
    try:
        result = await user_llm.generate_text(
            model=settings.handler_model,
            contents=prompt,
            system_instruction=(
                "You simulate a recruiter using an SHL assessment recommender. "
                "Output only the next user message."
            ),
            temperature=0.2,
            top_p=settings.top_p,
            max_output_tokens=160,
        )
    except Exception:
        if next_user_idx >= len(persona.user_turns):
            return "Locking it in. Thanks.", next_user_idx
        return persona.user_turns[next_user_idx], next_user_idx + 1

    text = (result.text or "").strip().strip('"')
    if not text:
        text = "Locking it in. Thanks."
    return text.splitlines()[0][:800], next_user_idx


def _simulated_user_prompt(persona: SamplePersona, messages: list[Message]) -> str:
    transcript = "\n".join(f"{m.role}: {m.content}" for m in messages[-6:])
    return (
        "PERSONA FACTS:\n"
        f"{persona.fact_summary}\n\n"
        "EXPECTED NEEDS (assessment names are hidden from the assistant, use only as your private goal):\n"
        f"{', '.join(persona.expected_names)}\n\n"
        "CONVERSATION SO FAR:\n"
        f"{transcript}\n\n"
        "Write the next user reply. Answer clarifying questions truthfully from the facts. "
        "Say 'no preference' for facts you do not know. If the assistant has provided a "
        "reasonable shortlist, confirm briefly. Do not introduce requirements outside the facts."
    )


def write_jsonl(report: ReplayReport, out_dir: Path) -> Path:
    """Persist the report as JSONL — one line per trace, plus a summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"replay_{int(time.time())}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"summary": {"mean_recall_at_10": report.mean_recall_at_10, "schema_pass_rate": report.schema_pass_rate}}) + "\n")
        for t in report.traces:
            fh.write(json.dumps({**asdict(t)}) + "\n")
    return out_path
