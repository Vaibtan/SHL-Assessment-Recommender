"""Tool declarations + Python implementations for the planning layer.

Tools are exposed to handlers selectively (curated toolkit per intent).
Each tool has:
- a `FunctionDeclaration` for Gemini's function-calling API
- a Python `_impl` callable invoked when Gemini emits a function call
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import structlog
from google.genai import types as gtypes

from shl_recommender.catalog.loader import CatalogIndex
from shl_recommender.catalog.normalize import CatalogItem
from shl_recommender.catalog.retrieval import RetrievalFilters

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class ToolBox:
    """A typed bundle of tool implementations bound to a CatalogIndex.

    Handlers select a subset of these to expose to Gemini per-call.
    """

    index: CatalogIndex
    query_vec_provider: Callable[[str], np.ndarray | None] | None = None

    # ----------------------------- search_catalog -----------------------------

    def search_catalog(
        self,
        query: str,
        test_types: list[str] | None = None,
        languages: list[str] | None = None,
        job_levels: list[str] | None = None,
        duration_max_minutes: int | None = None,
        remote_only: bool = False,
        adaptive_only: bool = False,
        coverage_letters: list[str] | None = None,
        top_k: int = 12,
    ) -> dict[str, Any]:
        filters = RetrievalFilters(
            test_types=tuple(test_types or ()),
            languages=tuple(languages or ()),
            job_levels=tuple(job_levels or ()),
            duration_max_minutes=duration_max_minutes,
            remote_only=remote_only,
            adaptive_only=adaptive_only,
        )
        query_vec = self.query_vec_provider(query) if self.query_vec_provider else None
        retriever = self.index.retriever
        hits = retriever.retrieve(
            query=query,
            query_vec=query_vec,
            filters=filters,
            coverage_letters=tuple(coverage_letters or ()),
            final_k=top_k,
        )
        return {"results": [_hit_summary(h, self.index.get(h.entity_id)) for h in hits]}

    # ----------------------------- get_assessment -----------------------------

    def get_assessment(
        self,
        entity_id: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        item = None
        if entity_id:
            item = self.index.get(entity_id)
        if item is None and name:
            item = _resolve_by_name(name, self.index.items)
        if item is None:
            return {"found": False}
        return {"found": True, "item": _full_summary(item)}

    # ------------------------------ find_similar ------------------------------

    def find_similar(self, entity_id: str, top_k: int = 5) -> dict[str, Any]:
        retriever = self.index.retriever
        hits = retriever.find_similar(entity_id, k=top_k)
        return {"results": [_hit_summary(h, self.index.get(h.entity_id)) for h in hits]}

    # ------------------------------- list_facets ------------------------------

    def list_facets(self, query: str, top_k: int = 30) -> dict[str, Any]:
        """Return distinct facet values for items matching `query` — disambiguation aid."""
        query_vec = self.query_vec_provider(query) if self.query_vec_provider else None
        hits = self.index.retriever.retrieve(query=query, query_vec=query_vec, final_k=top_k)
        languages: set[str] = set()
        job_levels: set[str] = set()
        test_types: set[str] = set()
        for h in hits:
            it = self.index.get(h.entity_id)
            if it is None:
                continue
            languages.update(it.languages)
            job_levels.update(it.job_levels)
            for letter in it.test_type.split(","):
                test_types.add(letter)
        return {
            "languages": sorted(languages),
            "job_levels": sorted(job_levels),
            "test_types": sorted(test_types),
        }


# ----------------------------- declarations -----------------------------------

# Filter-shaped properties shared across declarations to keep declarations DRY.
_FILTER_PROPS: dict[str, dict[str, Any]] = {
    "test_types": {
        "type": "ARRAY",
        "items": {"type": "STRING"},
        "description": "any-of: K|P|A|S|B|C|D|E (single-letter test type codes)",
    },
    "languages": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "any-of language substrings"},
    "job_levels": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "any-of job-level substrings"},
    "duration_max_minutes": {"type": "INTEGER"},
    "remote_only": {"type": "BOOLEAN"},
    "adaptive_only": {"type": "BOOLEAN"},
    "coverage_letters": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "K|P|A|S|B|C|D|E categories whose exemplar should be guaranteed in the candidate pool"},
}


SEARCH_CATALOG_DECL = gtypes.FunctionDeclaration(
    name="search_catalog",
    description=(
        "Hybrid BM25 + dense retrieval over the SHL catalog with optional facet "
        "filters and category-coverage injection. Returns up to top_k items with "
        "name, test_type, keys, duration, languages, and a description snippet."
    ),
    parameters=gtypes.Schema(
        type="OBJECT",
        properties={
            "query": gtypes.Schema(type="STRING", description="free-text query"),
            **{k: gtypes.Schema(**v) for k, v in _FILTER_PROPS.items()},
            "top_k": gtypes.Schema(type="INTEGER"),
        },
        required=["query"],
    ),
)

GET_ASSESSMENT_DECL = gtypes.FunctionDeclaration(
    name="get_assessment",
    description="Fetch a single SHL assessment by entity_id (preferred) or name (fuzzy).",
    parameters=gtypes.Schema(
        type="OBJECT",
        properties={
            "entity_id": gtypes.Schema(type="STRING"),
            "name": gtypes.Schema(type="STRING"),
        },
    ),
)

FIND_SIMILAR_DECL = gtypes.FunctionDeclaration(
    name="find_similar",
    description="Embedding-neighbor search starting from an existing entity_id.",
    parameters=gtypes.Schema(
        type="OBJECT",
        properties={
            "entity_id": gtypes.Schema(type="STRING"),
            "top_k": gtypes.Schema(type="INTEGER"),
        },
        required=["entity_id"],
    ),
)

LIST_FACETS_DECL = gtypes.FunctionDeclaration(
    name="list_facets",
    description="Distinct facet values (languages, job levels, test types) for items matching a query — disambiguation aid.",
    parameters=gtypes.Schema(
        type="OBJECT",
        properties={
            "query": gtypes.Schema(type="STRING"),
            "top_k": gtypes.Schema(type="INTEGER"),
        },
        required=["query"],
    ),
)


def make_tools(decls: list[gtypes.FunctionDeclaration]) -> list[gtypes.Tool]:
    """Wrap declarations into a single Tool object (Gemini's preferred shape)."""
    return [gtypes.Tool(function_declarations=list(decls))]


def dispatch(toolbox: ToolBox, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Invoke the named tool with kwargs; never raises (returns {"error": ...} instead)."""
    try:
        if name == "search_catalog":
            return toolbox.search_catalog(**args)
        if name == "get_assessment":
            return toolbox.get_assessment(**args)
        if name == "find_similar":
            return toolbox.find_similar(**args)
        if name == "list_facets":
            return toolbox.list_facets(**args)
        return {"error": f"unknown tool: {name}"}
    except TypeError as e:  # bad args from model
        log.warning("tool_bad_args", tool=name, error=str(e), args=args)
        return {"error": f"bad arguments: {e}"}
    except Exception as e:  # pragma: no cover - defensive
        log.exception("tool_unhandled_error", tool=name, error=str(e))
        return {"error": "internal tool error"}


# ----------------------------- helpers ----------------------------------------


def _hit_summary(hit, item: CatalogItem | None) -> dict[str, Any]:
    if item is None:
        return {"entity_id": hit.entity_id, "score": hit.score, "found": False}
    return {
        "entity_id": item.entity_id,
        "name": item.name,
        "test_type": item.test_type,
        "keys": list(item.keys),
        "job_levels": list(item.job_levels),
        "languages": list(item.languages),
        "duration": item.duration,
        "score": round(hit.score, 4),
        "snippet": _snippet(item.description, 200),
        "injected": hit.injected,
    }


def _full_summary(item: CatalogItem) -> dict[str, Any]:
    return {
        "entity_id": item.entity_id,
        "name": item.name,
        "url": item.url,
        "test_type": item.test_type,
        "keys": list(item.keys),
        "job_levels": list(item.job_levels),
        "languages": list(item.languages),
        "duration": item.duration,
        "duration_minutes": item.duration_minutes,
        "remote": item.remote,
        "adaptive": item.adaptive,
        "description": item.description,
    }


def _snippet(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"


def _resolve_by_name(name: str, items: list[CatalogItem]) -> CatalogItem | None:
    """Exact match first, then fuzzy via rapidfuzz."""
    target = name.strip().lower()
    if not target:
        return None
    for it in items:
        if it.name.lower() == target:
            return it
    # Fuzzy
    from rapidfuzz import fuzz, process

    choices = {it.entity_id: it.name for it in items}
    match = process.extractOne(name, choices, scorer=fuzz.WRatio, score_cutoff=85)
    if match is None:
        return None
    matched_name, score, key = match
    return next((it for it in items if it.entity_id == key), None)
