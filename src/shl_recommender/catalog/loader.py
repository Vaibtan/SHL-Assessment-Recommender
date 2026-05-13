# Purpose: CatalogIndex — singleton loader composed of items, BM25, embeddings, coverage.

from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process

from shl_recommender.config import get_settings
from shl_recommender.catalog.normalize import CatalogItem, from_parquet_records
from shl_recommender.catalog.retrieval import CategoryCoverage, Retriever, l2_normalize, tokenize

CATALOG_PARQUET: Final[str] = "catalog.parquet"
EMBEDDINGS_NPY: Final[str] = "embeddings.npy"
BM25_PKL: Final[str] = "bm25_index.pkl"
COVERAGE_PKL: Final[str] = "coverage.pkl"
META_JSON: Final[str] = "meta.json"


@dataclass(frozen=True, slots=True)
class CatalogIndex:
    """Immutable in-memory catalog facade — created once, shared across requests.

    The public interface deliberately owns lookup maps and the Retriever so
    request-path code does not rebuild O(N) structures per turn.
    """

    items: Sequence[CatalogItem]
    bm25: BM25Okapi
    embeddings: np.ndarray  # L2-normalized (N, D)
    coverage: CategoryCoverage
    _by_id: dict[str, CatalogItem] = field(init=False, repr=False, compare=False)
    _by_url: dict[str, CatalogItem] = field(init=False, repr=False, compare=False)
    _by_name_lc: dict[str, CatalogItem] = field(init=False, repr=False, compare=False)
    _name_choices: dict[str, str] = field(init=False, repr=False, compare=False)
    _retriever: Retriever = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "_by_id", {it.entity_id: it for it in self.items})
        object.__setattr__(self, "_by_url", {it.url.rstrip("/"): it for it in self.items})
        object.__setattr__(self, "_by_name_lc", {it.name.lower(): it for it in self.items})
        object.__setattr__(self, "_name_choices", {it.entity_id: it.name for it in self.items})
        object.__setattr__(
            self,
            "_retriever",
            Retriever(self.items, self.bm25, self.embeddings, self.coverage),
        )

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    def get(self, entity_id: str) -> CatalogItem | None:
        return self._by_id.get(entity_id)

    def get_by_url(self, url: str) -> CatalogItem | None:
        return self._by_url.get(url.rstrip("/"))

    def get_by_name(self, name: str) -> CatalogItem | None:
        return self._by_name_lc.get(name.strip().lower())

    def resolve_name(self, name: str, *, score_cutoff: int = 85) -> CatalogItem | None:
        """Resolve a catalog item by exact name, then RapidFuzz fallback."""
        exact = self.get_by_name(name)
        if exact is not None:
            return exact
        target = name.strip()
        if not target:
            return None
        match = process.extractOne(
            target,
            self._name_choices,
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff,
        )
        if match is None:
            return None
        _, _, entity_id = match
        return self.get(str(entity_id))

    def suggest_name(self, name: str, *, score_cutoff: int = 70) -> str | None:
        """Return the closest catalog item name, or None when nothing is close."""
        target = name.strip()
        if not target:
            return None
        match = process.extractOne(
            target,
            self._name_choices,
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff,
        )
        if match is None:
            return None
        matched_name, _, _ = match
        return str(matched_name)

    def items_for_ids(self, entity_ids: Iterable[str]) -> list[CatalogItem]:
        """Resolve IDs to items, preserving order and dropping unknown IDs."""
        return [item for eid in entity_ids if (item := self.get(eid)) is not None]


def save_index(
    out_dir: Path,
    items: Sequence[CatalogItem],
    embeddings: np.ndarray,
    bm25: BM25Okapi,
    coverage: CategoryCoverage,
) -> None:
    """Write all index artifacts to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(_records(items))
    df.to_parquet(out_dir / CATALOG_PARQUET, index=False)
    np.save(out_dir / EMBEDDINGS_NPY, embeddings.astype(np.float32))
    with open(out_dir / BM25_PKL, "wb") as fh:
        pickle.dump(bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_dir / COVERAGE_PKL, "wb") as fh:
        pickle.dump(coverage, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(in_dir: Path) -> CatalogIndex:
    """Read all index artifacts from `in_dir` and rebuild a CatalogIndex."""
    df = pd.read_parquet(in_dir / CATALOG_PARQUET)
    items = from_parquet_records(df.to_dict(orient="records"))
    embeddings = np.load(in_dir / EMBEDDINGS_NPY).astype(np.float32)
    _validate_embeddings_shape(embeddings, items, in_dir)
    if not _is_normalized(embeddings):
        embeddings = l2_normalize(embeddings)
    with open(in_dir / BM25_PKL, "rb") as fh:
        bm25 = pickle.load(fh)
    with open(in_dir / COVERAGE_PKL, "rb") as fh:
        coverage = pickle.load(fh)
    return CatalogIndex(items=items, bm25=bm25, embeddings=embeddings, coverage=coverage)


def _validate_embeddings_shape(
    embeddings: np.ndarray,
    items: Sequence[CatalogItem],
    in_dir: Path,
) -> None:
    """Fail fast when runtime embedding config and baked artifacts diverge."""
    if embeddings.ndim != 2:
        raise ValueError(f"{EMBEDDINGS_NPY} must be a 2D matrix, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(items):
        raise ValueError(
            f"{EMBEDDINGS_NPY} row count {embeddings.shape[0]} does not match "
            f"{CATALOG_PARQUET} item count {len(items)} in {in_dir}"
        )
    expected_dims = get_settings().embedding_dims
    if embeddings.shape[1] != expected_dims:
        raise ValueError(
            f"{EMBEDDINGS_NPY} has {embeddings.shape[1]} dimensions, but "
            f"SHL_EMBEDDING_DIMS is {expected_dims}. Rebuild the index artifacts "
            "or remove the runtime override."
        )


def build_bm25(items: Sequence[CatalogItem], search_text_fn) -> BM25Okapi:
    """Build a BM25 index from the items' search-text representations."""
    corpus = [tokenize(search_text_fn(it)) for it in items]
    return BM25Okapi(corpus)


def derive_default_coverage(items: Sequence[CatalogItem]) -> CategoryCoverage:
    """Pick exemplar items per test_type letter for category-coverage injection.

    Strategy: prefer well-known SHL anchors when present (OPQ32r, Verify G+,
    Graduate Scenarios, DSI, GSA), then fall back to whichever single-letter
    items have the longest descriptions (a proxy for "core" assessments).
    """
    by_letter: dict[str, list[tuple[int, str]]] = {}
    canonical_anchors: dict[str, list[str]] = {
        "P": ["occupational personality questionnaire opq32r", "dependability and safety instrument"],
        "A": ["shl verify interactive g+", "shl verify interactive g"],
        "B": ["graduate scenarios"],
        "K": ["global skills assessment"],
        "C": ["global skills assessment"],
        "D": ["global skills development report"],
        "S": ["contact center call simulation"],
    }

    name_to_id: dict[str, str] = {it.name.lower(): it.entity_id for it in items}

    exemplars: dict[str, list[str]] = {letter: [] for letter in "KPASBCDE"}

    for letter, names in canonical_anchors.items():
        for name in names:
            for cand_name, eid in name_to_id.items():
                if name in cand_name and eid not in exemplars[letter]:
                    exemplars[letter].append(eid)

    for letter in "KPASBCDE":
        if len(exemplars[letter]) >= 5:
            continue
        candidates = [
            (it.entity_id, len(it.description))
            for it in items
            if letter in it.test_type.split(",") and it.entity_id not in exemplars[letter]
        ]
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        for eid, _ in candidates[: 5 - len(exemplars[letter])]:
            exemplars[letter].append(eid)

    return CategoryCoverage(exemplars={k: tuple(v) for k, v in exemplars.items()})


def _records(items: Sequence[CatalogItem]) -> list[dict]:
    """Inline import-free serializer (same shape as normalize.to_parquet_records)."""
    from shl_recommender.catalog.normalize import to_parquet_records  # local to avoid cycle in tests

    return to_parquet_records(items)


def _is_normalized(matrix: np.ndarray, tol: float = 1e-3) -> bool:
    if matrix.size == 0:
        return True
    norms = np.linalg.norm(matrix, axis=1)
    return bool(np.all(np.abs(norms - 1.0) < tol))
