# Purpose: Retrieval — hybrid BM25 + dense + RRF, hard facet filters, category coverage.

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final, Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from shl_recommender.catalog.normalize import CatalogItem

RRF_K: Final[int] = 60

DEFAULT_PER_RETRIEVER_K: Final[int] = 30

_TOKEN_RE = re.compile(r"\.?[A-Za-z0-9][A-Za-z0-9\-+#.]*")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens, preserving in-word `-`, `+`, `#`, `.`.

    These characters matter for our domain — "C++", ".NET", "Verify G+",
    "Java 8", "OPQ32r" should each be one token, not split. Trailing
    punctuation is trimmed.
    """
    if not text:
        return []
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        while tok.endswith("."):
            tok = tok[:-1]
        if tok:
            out.append(tok)
    return out


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """Hard pre-filters applied before scoring."""

    test_types: tuple[str, ...] = ()  # any-of: e.g. ("K", "P")
    languages: tuple[str, ...] = ()  # any-of substring match
    job_levels: tuple[str, ...] = ()  # any-of substring match
    duration_max_minutes: int | None = None
    remote_only: bool = False
    adaptive_only: bool = False

    def is_empty(self) -> bool:
        return not (
            self.test_types
            or self.languages
            or self.job_levels
            or self.duration_max_minutes is not None
            or self.remote_only
            or self.adaptive_only
        )


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A single retrieval result with provenance."""

    entity_id: str
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    injected: bool = False  # True if added via category-coverage injection


@dataclass(frozen=True, slots=True)
class CategoryCoverage:
    """Defines which test_type categories should be guaranteed in the candidate pool.

    For each category letter, we keep a small ranked list of "exemplar" entity_ids
    that get injected when retrieval underrepresents that category. Exemplars are
    chosen at index-build time from the most-tagged / canonical items per category.
    """

    exemplars: dict[str, tuple[str, ...]]  # letter -> (entity_id, ...)


def _matches_filters(item: CatalogItem, filters: RetrievalFilters) -> bool:
    """Return True iff item satisfies every filter."""
    if filters.test_types:
        item_letters = set(item.test_type.split(","))
        if not item_letters.intersection(filters.test_types):
            return False
    if filters.languages:
        item_langs_lc = [l.lower() for l in item.languages]
        if not any(any(f.lower() in il for il in item_langs_lc) for f in filters.languages):
            return False
    if filters.job_levels:
        item_jls_lc = [j.lower() for j in item.job_levels]
        if not any(any(f.lower() in ij for ij in item_jls_lc) for f in filters.job_levels):
            return False
    if filters.duration_max_minutes is not None:
        if item.duration_minutes is None:
            return False
        if item.duration_minutes > filters.duration_max_minutes:
            return False
    if filters.remote_only and not item.remote:
        return False
    if filters.adaptive_only and not item.adaptive:
        return False
    return True


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists by reciprocal-rank-fusion.

    rankings: each inner sequence is an ordered list of entity_ids (rank 0 = best).
    Returns ids sorted by descending fused score.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, eid in enumerate(ranking):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def cosine_topk(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    k: int,
    candidate_idx: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Top-K cosine similarity. Vectors assumed L2-normalized.

    candidate_idx: optional restriction to a subset of row indices.
    Returns [(matrix_row_index, score), ...] descending.
    """
    if candidate_idx is not None:
        if candidate_idx.size == 0:
            return []
        sub = matrix[candidate_idx]
        sims = sub @ query_vec
        if sims.size <= k:
            order = np.argsort(-sims)
        else:
            top_unsorted = np.argpartition(-sims, k - 1)[:k]
            order = top_unsorted[np.argsort(-sims[top_unsorted])]
        return [(int(candidate_idx[i]), float(sims[i])) for i in order]
    sims = matrix @ query_vec
    if sims.size <= k:
        order = np.argsort(-sims)
    else:
        top_unsorted = np.argpartition(-sims, k - 1)[:k]
        order = top_unsorted[np.argsort(-sims[top_unsorted])]
    return [(int(i), float(sims[i])) for i in order]


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


class Retriever:
    """Stateless retriever over a CatalogIndex.

    Given a query (text + filters), returns a fused candidate pool with
    optional category-coverage injection.
    """

    def __init__(
        self,
        items: Sequence[CatalogItem],
        bm25: BM25Okapi,
        embeddings: np.ndarray,
        category_coverage: CategoryCoverage,
    ) -> None:
        self.items = items
        self.bm25 = bm25
        self.embeddings = embeddings  # L2-normalized (N, D)
        self.coverage = category_coverage
        self._id_to_idx: dict[str, int] = {it.entity_id: i for i, it in enumerate(items)}

    @property
    def size(self) -> int:
        return len(self.items)

    def _bm25_rank(self, query: str, candidate_idx: np.ndarray | None, k: int) -> list[int]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if candidate_idx is not None:
            mask = np.full(scores.shape, -np.inf, dtype=scores.dtype)
            mask[candidate_idx] = scores[candidate_idx]
            scores = mask
        if scores.size <= k:
            order = np.argsort(-scores)
        else:
            top_unsorted = np.argpartition(-scores, k - 1)[:k]
            order = top_unsorted[np.argsort(-scores[top_unsorted])]
        return [int(i) for i in order if math.isfinite(float(scores[i]))]

    def _dense_rank(self, query_vec: np.ndarray | None, candidate_idx: np.ndarray | None, k: int) -> list[int]:
        if query_vec is None:
            return []
        results = cosine_topk(query_vec, self.embeddings, k=k, candidate_idx=candidate_idx)
        return [idx for idx, _ in results]

    def _filter_indices(self, filters: RetrievalFilters) -> np.ndarray | None:
        if filters.is_empty():
            return None
        keep = [i for i, it in enumerate(self.items) if _matches_filters(it, filters)]
        return np.asarray(keep, dtype=np.int64) if keep else np.asarray([], dtype=np.int64)

    def retrieve(
        self,
        query: str,
        query_vec: np.ndarray | None = None,
        filters: RetrievalFilters | None = None,
        per_retriever_k: int = DEFAULT_PER_RETRIEVER_K,
        final_k: int = 30,
        coverage_letters: Iterable[str] = (),
    ) -> list[RetrievalHit]:
        """Run hybrid retrieval, RRF fusion, and category-coverage injection.

        Args:
            query: free-text query for BM25 (and embedded query vector for dense).
            query_vec: pre-computed L2-normalized query embedding (optional).
            filters: hard pre-filters (None = no filtering).
            per_retriever_k: top-K per retriever before fusion.
            final_k: cap on the returned candidate pool.
            coverage_letters: test_type letters whose exemplars should be present
                in the pool. Missing ones are injected at the tail.

        Returns:
            Ordered list of RetrievalHit. If hard filtering reduces the pool to
            empty, returns []; the caller is responsible for relaxing filters.
        """
        filters = filters or RetrievalFilters()
        candidate_idx = self._filter_indices(filters)
        if candidate_idx is not None and candidate_idx.size == 0:
            return []

        bm25_idx = self._bm25_rank(query, candidate_idx, per_retriever_k)
        dense_idx = self._dense_rank(query_vec, candidate_idx, per_retriever_k)

        bm25_ids = [self.items[i].entity_id for i in bm25_idx]
        dense_ids = [self.items[i].entity_id for i in dense_idx]

        fused = reciprocal_rank_fusion([bm25_ids, dense_ids])
        bm25_rank_map = {eid: r for r, eid in enumerate(bm25_ids)}
        dense_rank_map = {eid: r for r, eid in enumerate(dense_ids)}

        hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for eid, score in fused[:final_k]:
            hits.append(
                RetrievalHit(
                    entity_id=eid,
                    score=score,
                    bm25_rank=bm25_rank_map.get(eid),
                    dense_rank=dense_rank_map.get(eid),
                )
            )
            seen.add(eid)

        for letter in coverage_letters:
            if any(letter in self.items[self._id_to_idx[h.entity_id]].test_type.split(",") for h in hits):
                continue
            for exemplar_id in self.coverage.exemplars.get(letter, ()):
                if exemplar_id in seen:
                    break
                idx = self._id_to_idx.get(exemplar_id)
                if idx is None:
                    continue
                if candidate_idx is not None and not _matches_filters(self.items[idx], filters):
                    continue
                hits.append(
                    RetrievalHit(
                        entity_id=exemplar_id,
                        score=0.0,
                        bm25_rank=None,
                        dense_rank=None,
                        injected=True,
                    )
                )
                seen.add(exemplar_id)
                break

        return hits

    def find_similar(self, entity_id: str, k: int = 5) -> list[RetrievalHit]:
        """Embedding-neighbor lookup (excludes the source item)."""
        idx = self._id_to_idx.get(entity_id)
        if idx is None:
            return []
        query_vec = self.embeddings[idx]
        candidate_idx = np.asarray([i for i in range(self.size) if i != idx], dtype=np.int64)
        results = cosine_topk(query_vec, self.embeddings, k=k, candidate_idx=candidate_idx)
        return [
            RetrievalHit(
                entity_id=self.items[i].entity_id,
                score=score,
                bm25_rank=None,
                dense_rank=rank,
            )
            for rank, (i, score) in enumerate(results)
        ]
