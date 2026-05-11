"""High-precision catalog query expansion.

This module is the deterministic companion to hybrid retrieval. It promotes
catalog items that are strongly implied by exact skills, product acronyms, or
stable SHL-domain synonyms before an LLM selects from the closed candidate pool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final, Iterable, Sequence

from shl_recommender.catalog.normalize import CatalogItem


@dataclass(frozen=True, slots=True)
class ExpandedCandidates:
    """Catalog candidates plus diagnostics for observability."""

    items: tuple[CatalogItem, ...]
    matched_aliases: tuple[str, ...] = ()
    matched_concepts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _AliasEntry:
    alias: str
    entity_id: str
    score: float
    source: str


@dataclass(frozen=True, slots=True)
class _ConceptRule:
    name: str
    triggers: tuple[str, ...]
    targets: tuple[str, ...]


@dataclass(slots=True)
class _Hit:
    item: CatalogItem
    score: float
    first_order: int
    matched_aliases: set[str] = field(default_factory=set)
    matched_concepts: set[str] = field(default_factory=set)


_GENERIC_NAME_TOKENS: Final[set[str]] = {
    "assessment",
    "candidate",
    "candidates",
    "development",
    "general",
    "interactive",
    "level",
    "new",
    "questionnaire",
    "report",
    "reports",
    "solution",
    "solutions",
    "test",
    "verify",
}

_HIGH_SIGNAL_SINGLE_TOKENS: Final[set[str]] = {
    "aws",
    "docker",
    "excel",
    "hipaa",
    "java",
    "linux",
    "numerical",
    "sales",
    "spring",
    "sql",
    "word",
}

# Domain-level synonyms. These are deliberately phrased as reusable hiring and
# assessment concepts rather than replay-sample branches.
_CONCEPT_RULES: Final[tuple[_ConceptRule, ...]] = (
    _ConceptRule(
        name="leadership_personality",
        triggers=(
            r"\bleadership\b",
            r"\bexecutive\b",
            r"\bcxo\b",
            r"\bdirector[- ]level\b",
            r"\bbenchmark\b",
        ),
        targets=(
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ Universal Competency Report 2.0",
            "OPQ Leadership Report",
        ),
    ),
    _ConceptRule(
        name="systems_programming",
        triggers=(
            r"\brust\b",
            r"\blinux\b",
            r"\bnetworking\b",
            r"\bhigh[- ]performance\b",
            r"\binfrastructure\b",
        ),
        targets=(
            "Smart Interview Live Coding",
            "Linux Programming (General)",
            "Networking and Implementation",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ),
    ),
    _ConceptRule(
        name="contact_center",
        triggers=(
            r"\bcontact cent(?:er|re)\b",
            r"\bcustomer service\b",
            r"\binbound calls?\b",
            r"\bspoken english\b",
        ),
        targets=(
            "SVAR - Spoken English (US)",
            "Contact Center Call Simulation",
            "Entry Level Customer Serv-Retail & Contact Center",
            "Customer Service Phone Simulation",
        ),
    ),
    _ConceptRule(
        name="finance_numeracy",
        triggers=(
            r"\bfinancial analysts?\b",
            r"\bfinance\b",
            r"\bfinancial accounting\b",
            r"\bnumerical\b",
            r"\bbasic statistics\b",
        ),
        targets=(
            "SHL Verify Interactive – Numerical Reasoning",
            "Financial Accounting",
            "Basic Statistics",
            "Graduate Scenarios",
            "Occupational Personality Questionnaire OPQ32r",
        ),
    ),
    _ConceptRule(
        name="sales_capability",
        triggers=(
            r"\bsales\b",
            r"\breskill\b",
            r"\bre-skill\b",
            r"\bcapability audit\b",
            r"\btransformation\b",
        ),
        targets=(
            "Global Skills Assessment",
            "Global Skills Development Report",
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ MQ Sales Report",
            "Sales Transformation 2.0 - Individual Contributor",
        ),
    ),
    _ConceptRule(
        name="healthcare_administration",
        triggers=(
            r"\bhipaa\b",
            r"\bhealthcare\b",
            r"\bpatient records?\b",
            r"\bmedical terminology\b",
            r"\bbilingual\b",
        ),
        targets=(
            "HIPAA (Security)",
            "Medical Terminology",
            "Microsoft Word 365 - Essentials",
            "Dependability and Safety Instrument (DSI)",
            "Occupational Personality Questionnaire OPQ32r",
        ),
    ),
    _ConceptRule(
        name="office_productivity",
        triggers=(
            r"\bmicrosoft office\b",
            r"\boffice 365\b",
            r"\bmicrosoft 365\b",
            r"\bexcel\b",
            r"\bword\b",
            r"\badmin assistants?\b",
        ),
        targets=(
            "Microsoft 365",
            "Microsoft Word 365",
            "MS Excel",
            "MS Word",
            "Occupational Personality Questionnaire OPQ32r",
        ),
    ),
    _ConceptRule(
        name="software_engineering_stack",
        triggers=(
            r"\bcore java\b",
            r"\bspring\b",
            r"\bsql\b",
            r"\baws\b",
            r"\bdocker\b",
            r"\bfull[- ]stack\b",
            r"\bmicroservices?\b",
        ),
        targets=(
            "Core Java (Advanced Level)",
            "Spring",
            "SQL",
            "Amazon Web Services (AWS) Development",
            "Docker",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ),
    ),
    _ConceptRule(
        name="industrial_safety",
        triggers=(
            r"\bplant operators?\b",
            r"\bchemical facilit(?:y|ies)\b",
            r"\bprocedure compliance\b",
            r"\bworkplace health and safety\b",
            r"\bsafety\b",
        ),
        targets=(
            "Manufac. & Indust. - Safety & Dependability 8.0",
            "Workplace Health and Safety",
            "Verify Interactive Process Monitoring",
            "Dependability and Safety Instrument (DSI)",
        ),
    ),
    _ConceptRule(
        name="graduate_hiring",
        triggers=(
            r"\bgraduate management trainee\b",
            r"\bmanagement trainee\b",
            r"\bgraduate scenarios?\b",
            r"\bgraduate\b",
        ),
        targets=(
            "SHL Verify Interactive G+",
            "Graduate Scenarios",
            "Occupational Personality Questionnaire OPQ32r",
        ),
    ),
)


def expand_catalog_candidates(
    query_text: str,
    items: Sequence[CatalogItem],
    *,
    limit: int = 12,
) -> ExpandedCandidates:
    """Return deterministic, high-confidence catalog candidates.

    Expansion is intentionally conservative: it only emits existing catalog
    items, deduplicates by entity_id, and orders concept matches before broader
    catalog-name aliases.
    """
    normalized_query = normalize_text(query_text)
    if not normalized_query or not items:
        return ExpandedCandidates(items=())

    by_id = {it.entity_id: it for it in items}
    hits: dict[str, _Hit] = {}
    order = 0

    for rule in _CONCEPT_RULES:
        if not _rule_matches(rule, query_text):
            continue
        for target_rank, target in enumerate(rule.targets):
            item = _resolve_item_by_name(target, items)
            if item is None:
                continue
            order = _add_hit(
                hits,
                item,
                score=100.0 - target_rank,
                order=order,
                matched_concept=rule.name,
            )

    for entry in _alias_entries(tuple((it.entity_id, it.name) for it in items)):
        if entry.entity_id not in by_id:
            continue
        if _contains_phrase(normalized_query, entry.alias):
            item = by_id[entry.entity_id]
            score = _score_alias_match(entry, item, normalized_query)
            order = _add_hit(
                hits,
                item,
                score=score,
                order=order,
                matched_alias=entry.alias,
            )

    ranked = sorted(hits.values(), key=lambda hit: (-hit.score, hit.first_order, hit.item.name))
    selected = ranked[:limit]
    aliases = _sorted_unique(alias for hit in selected for alias in hit.matched_aliases)
    concepts = _sorted_unique(concept for hit in selected for concept in hit.matched_concepts)
    return ExpandedCandidates(
        items=tuple(hit.item for hit in selected),
        matched_aliases=tuple(aliases),
        matched_concepts=tuple(concepts),
    )


def normalize_text(value: str) -> str:
    """Normalize text for word-boundary phrase matching."""
    return re.sub(r"[^a-z0-9+]+", " ", value.lower()).strip()


def _rule_matches(rule: _ConceptRule, query_text: str) -> bool:
    return any(re.search(trigger, query_text, flags=re.IGNORECASE) for trigger in rule.triggers)


def _add_hit(
    hits: dict[str, _Hit],
    item: CatalogItem,
    *,
    score: float,
    order: int,
    matched_alias: str | None = None,
    matched_concept: str | None = None,
) -> int:
    existing = hits.get(item.entity_id)
    if existing is None:
        existing = _Hit(item=item, score=score, first_order=order)
        hits[item.entity_id] = existing
        order += 1
    elif score > existing.score:
        existing.score = score
    if matched_alias:
        existing.matched_aliases.add(matched_alias)
    if matched_concept:
        existing.matched_concepts.add(matched_concept)
    return order


def _score_alias_match(entry: _AliasEntry, item: CatalogItem, normalized_query: str) -> float:
    score = entry.score
    if "report" in item.name.lower() and not _contains_phrase(normalized_query, "report"):
        score -= 8.0
    return score


@lru_cache(maxsize=8)
def _alias_entries(item_signature: tuple[tuple[str, str], ...]) -> tuple[_AliasEntry, ...]:
    entries: list[_AliasEntry] = []
    seen: set[tuple[str, str]] = set()
    for entity_id, name in item_signature:
        for alias, score, source in _aliases_for_name(name):
            key = (entity_id, alias)
            if key in seen:
                continue
            entries.append(_AliasEntry(alias=alias, entity_id=entity_id, score=score, source=source))
            seen.add(key)
    return tuple(entries)


def _aliases_for_name(name: str) -> list[tuple[str, float, str]]:
    aliases: list[tuple[str, float, str]] = []
    normalized = normalize_text(_strip_new_marker(name))
    if normalized:
        aliases.append((normalized, 96.0, "full_name"))

    for paren in re.findall(r"\(([^)]+)\)", name):
        alias = normalize_text(paren)
        if _is_acronym(paren) or _is_meaningful_single_alias(alias):
            aliases.append((alias, 91.0, "parenthetical"))

    for raw_token in re.findall(r"\b[A-Z]{2,}[A-Za-z0-9+]*\b|(?<![A-Za-z0-9])[A-Z]\+(?![A-Za-z0-9])", name):
        alias = normalize_text(raw_token)
        if _is_acronym(raw_token) or _is_meaningful_single_alias(alias):
            aliases.append((alias, 90.0, "acronym"))
        prefix = re.match(r"([a-z]{2,})\d", alias)
        if prefix:
            aliases.append((prefix.group(1), 89.0, "acronym_prefix"))

    tokens = [tok for tok in normalize_text(name).split() if tok not in _GENERIC_NAME_TOKENS]
    for token in tokens:
        if _is_meaningful_single_alias(token):
            aliases.append((token, 84.0, "name_token"))

    for width in (4, 3, 2):
        for i in range(0, max(len(tokens) - width + 1, 0)):
            phrase = " ".join(tokens[i : i + width])
            if phrase:
                aliases.append((phrase, 88.0 + width, "name_phrase"))
    return aliases


def _strip_new_marker(name: str) -> str:
    return re.sub(r"\s*\(New\)\s*$", "", name, flags=re.IGNORECASE)


def _is_meaningful_single_alias(alias: str) -> bool:
    if not alias or " " in alias or alias in _GENERIC_NAME_TOKENS:
        return False
    return (
        alias in _HIGH_SIGNAL_SINGLE_TOKENS
        or bool(re.fullmatch(r"[a-z]+[0-9][a-z0-9]*", alias))
        or bool(re.fullmatch(r"[a-z]\+|[a-z]{2,}\+", alias))
    )


def _is_acronym(value: str) -> bool:
    token = value.strip()
    return bool(re.fullmatch(r"[A-Z0-9+]{2,}", token))


def _contains_phrase(normalized_text: str, normalized_phrase: str) -> bool:
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def _resolve_item_by_name(target: str, items: Sequence[CatalogItem]) -> CatalogItem | None:
    target_norm = normalize_text(target)
    exact = [it for it in items if normalize_text(_strip_new_marker(it.name)) == target_norm]
    if exact:
        return exact[0]
    starts = [it for it in items if normalize_text(it.name).startswith(target_norm)]
    if starts:
        return sorted(starts, key=lambda it: len(it.name))[0]
    contains = [it for it in items if target_norm in normalize_text(it.name)]
    if contains:
        return sorted(contains, key=lambda it: len(it.name))[0]
    return None


def _sorted_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        out.append(text)
        seen.add(text)
    return sorted(out)
