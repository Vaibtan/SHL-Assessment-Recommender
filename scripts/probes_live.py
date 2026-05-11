"""
Live behavior-probe runner — same probes as `tests/replay/probes.py` but
exercised against the real Gemini-backed agent.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT / ".env", override=False)

from shl_recommender.agent.llm import LLMClient
from shl_recommender.agent.runner import Agent
from shl_recommender.catalog.loader import load_index
from shl_recommender.observability.logging import configure_logging

# tests/ is at the project root, not inside src/. Add it to the path.
TESTS_DIR = ROOT
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from tests.replay.probes import run_probes


async def main() -> int:
    configure_logging("INFO")
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

    index = load_index(ROOT / "data" / "build")
    agent = Agent(index=index, llm=LLMClient())

    results = await run_probes(agent, index)
    passed = sum(1 for r in results if r.passed)
    print(f"--- Probe results: {passed}/{len(results)} passed ---")
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        print(f"  [{flag}] {r.name}: {r.detail}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
