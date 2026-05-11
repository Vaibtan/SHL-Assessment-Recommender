"""Live replay runner — exercises the full agent against real Vertex / AI Studio.

Usage:
    GOOGLE_CLOUD_PROJECT=<project> uv run python scripts/replay_live.py
    GOOGLE_API_KEY=<key>          uv run python scripts/replay_live.py

Loads the catalog index, builds a real LLMClient, replays every persona in
sample_conversations/, prints per-trace Recall@10 + summary, and writes a JSONL
artifact under data/replay_runs/.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT / ".env", override=False)

from shl_recommender.agent.llm import LLMClient
from shl_recommender.agent.runner import Agent
from shl_recommender.catalog.loader import load_index
from shl_recommender.observability.logging import configure_logging


async def main() -> int:
    configure_logging("INFO")
    index_dir = ROOT / "data" / "build"
    if not (index_dir / "catalog.parquet").exists():
        print("ERROR: index artifacts missing. Run scripts/build_index.py first.", file=sys.stderr)
        return 2

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GCP_API_KEY")
    )
    if not (project or api_key):
        print(
            "ERROR: set GOOGLE_CLOUD_PROJECT (Vertex) or GOOGLE_API_KEY/GEMINI_API_KEY/GCP_API_KEY (AI Studio).",
            file=sys.stderr,
        )
        return 2

    print(f"Loading index from {index_dir}")
    index = load_index(index_dir)
    print(f"  loaded {len(index.items)} items")

    llm = LLMClient()
    agent = Agent(index=index, llm=llm)

    # Local import to keep the CLI dep-light when the script is unused.
    from tests.replay.harness import replay_all, write_jsonl

    print("Replaying personas...")
    started = time.perf_counter()
    report = await replay_all(agent, user_llm=llm)
    elapsed = round(time.perf_counter() - started, 1)

    print()
    print(f"--- Replay summary (elapsed {elapsed}s) ---")
    print(f"  mean Recall@10 : {report.mean_recall_at_10}")
    print(f"  schema valid   : {report.schema_pass_rate}")
    for trace in report.traces:
        print(
            f"  {trace.sample_id}: recall={trace.recall_at_10:.2f} "
            f"end_reached={trace.end_reached} turns={len(trace.turns)} "
            f"predicted={len(trace.final_predicted_ids)}/{len(trace.expected_ids)}"
        )

    out_dir = ROOT / "data" / "replay_runs"
    out_path = write_jsonl(report, out_dir)
    print(f"\nWrote replay artifact to {out_path}")
    return 0 if report.mean_recall_at_10 >= 0.65 and report.schema_pass_rate == 1.0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
