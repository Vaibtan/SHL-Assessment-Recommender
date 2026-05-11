"""Persona extraction from sample_conversations/C*.md.

Each sample is a multi-turn dialogue where:
- The user's turn-1 message reveals the persona / scenario.
- Subsequent user turns add facts / preferences.
- The assistant's final-turn markdown table is the labeled expected shortlist.

We parse all 10 traces into structured `SamplePersona` records that the replay
harness can drive against the live agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz, process

from shl_recommender.catalog.loader import CatalogIndex


SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_conversations"

_TURN_RE = re.compile(r"^### Turn \d+\s*$", re.MULTILINE)
_USER_BLOCK_RE = re.compile(r"\*\*User\*\*\s*\n+(>.*?)(?=\n\n|\*\*Agent\*\*|$)", re.DOTALL)
_AGENT_BLOCK_RE = re.compile(r"\*\*Agent\*\*\s*\n+(.*?)(?=^### Turn|\Z)", re.DOTALL | re.MULTILINE)
_QUOTE_LINE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)
_TABLE_NAME_RE = re.compile(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|", re.MULTILINE)
_URL_RE = re.compile(r"<?(https?://www\.shl\.com/products/product-catalog/view/[^>\s|)]+)/?>?")


@dataclass(frozen=True, slots=True)
class SamplePersona:
    """A trace persona extracted from a C*.md file."""

    sample_id: str
    user_turns: tuple[str, ...]  # ordered user messages from the trace
    expected_entity_ids: tuple[str, ...]  # final shortlist (labels)
    expected_names: tuple[str, ...]  # raw names (for diagnostic logging)

    @property
    def opening_message(self) -> str:
        return self.user_turns[0] if self.user_turns else ""

    @property
    def fact_summary(self) -> str:
        """Best-effort facts summary the simulated user can answer from."""
        return " ".join(self.user_turns)


def load_all_personas(index: CatalogIndex) -> list[SamplePersona]:
    """Parse every C*.md trace and resolve names → entity_ids."""
    files = sorted(SAMPLE_DIR.glob("C*.md"), key=_natural_sort_key)
    return [parse_persona(p, index) for p in files]


def parse_persona(path: Path, index: CatalogIndex) -> SamplePersona:
    text = path.read_text(encoding="utf-8")
    user_turns = _extract_user_turns(text)
    expected_names, expected_ids = _extract_expected_shortlist(text, index)
    return SamplePersona(
        sample_id=path.stem,
        user_turns=tuple(user_turns),
        expected_entity_ids=tuple(expected_ids),
        expected_names=tuple(expected_names),
    )


def _extract_user_turns(text: str) -> list[str]:
    out: list[str] = []
    for m in _USER_BLOCK_RE.finditer(text):
        block = m.group(1)
        cleaned = _QUOTE_LINE_RE.sub("", block).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _extract_expected_shortlist(text: str, index: CatalogIndex) -> tuple[list[str], list[str]]:
    """The final assistant turn is the labeled shortlist; pull its table."""
    agent_blocks = _AGENT_BLOCK_RE.findall(text)
    if not agent_blocks:
        return [], []
    final_block = agent_blocks[-1]

    # Prefer URL-based resolution (most reliable).
    urls = _URL_RE.findall(final_block)
    by_url = {it.url.rstrip("/"): it.entity_id for it in index.items}
    by_name = {it.name: it.entity_id for it in index.items}

    ids: list[str] = []
    names: list[str] = []
    seen_ids: set[str] = set()

    for url in urls:
        eid = by_url.get(url.rstrip("/"))
        if eid and eid not in seen_ids:
            ids.append(eid)
            seen_ids.add(eid)

    if not ids:
        # Fallback: parse names from the table and fuzzy-match.
        candidate_names = [n.strip() for n in _TABLE_NAME_RE.findall(final_block)]
        for name in candidate_names:
            if not name or name.lower() == "name":
                continue
            eid = by_name.get(name)
            if eid is None:
                match = process.extractOne(name, by_name, scorer=fuzz.WRatio, score_cutoff=85)
                if match is not None:
                    eid = match[2]
            if eid and eid not in seen_ids:
                ids.append(eid)
                names.append(name)
                seen_ids.add(eid)
    else:
        # Map ids back to display names.
        id_to_name = {it.entity_id: it.name for it in index.items}
        names = [id_to_name.get(eid, "") for eid in ids]

    return names, ids


def _natural_sort_key(p: Path) -> tuple:
    """Sort C1.md, C2.md, ..., C10.md correctly."""
    digits = re.findall(r"\d+", p.stem)
    return tuple(int(d) for d in digits) if digits else (0,)
