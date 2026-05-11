"""Build script — produces all catalog index artifacts.

Usage:
    uv run python scripts/build_index.py [--dry-run]

Reads `data/shl_product_catalog.json`, normalizes records, computes
`gemini-embedding-001` vectors at 768 dims (Matryoshka), builds the BM25 index
and category-coverage exemplars, and writes `data/build/*` artifacts that the
runtime loads at FastAPI startup.

The script is idempotent — re-running overwrites the artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Ensure src/ is importable when running directly.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env from project root if present.
load_dotenv(dotenv_path=ROOT / ".env", override=False)

from shl_recommender.catalog.loader import (
    build_bm25,
    derive_default_coverage,
    save_index,
)
from shl_recommender.catalog.normalize import (
    CatalogItem,
    build_search_text,
    normalize_catalog,
)
from shl_recommender.catalog.retrieval import l2_normalize
from shl_recommender.config import get_settings

DEFAULT_INPUT: Path = ROOT / "data" / "shl_product_catalog.json"
DEFAULT_OUTPUT: Path = ROOT / "data" / "build"


def _genai_client():
    """Construct a Gemini client; honor Vertex credentials if available, else AI Studio."""
    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GCP_API_KEY")
    )

    if project and api_key:
        return genai.Client(vertexai=True, project=project, location=location, api_key=api_key)
    if project:
        return genai.Client(vertexai=True, project=project, location=location)
    if api_key:
        return genai.Client(api_key=api_key)
    raise RuntimeError(
        "Set GOOGLE_CLOUD_PROJECT (Vertex) or GOOGLE_API_KEY/GEMINI_API_KEY/GCP_API_KEY (AI Studio) to embed."
    )


def embed_corpus(texts: list[str]) -> np.ndarray:
    """Batch-embed text strings via the configured embedding model + dims."""
    from google.genai.types import EmbedContentConfig

    settings = get_settings()
    client = _genai_client()
    config = EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=settings.embedding_dims,
    )

    vectors: list[list[float]] = []
    batch_size = settings.embedding_batch_size
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        result = client.models.embed_content(
            model=settings.embedding_model,
            contents=chunk,
            config=config,
        )
        for emb in result.embeddings:
            vectors.append(list(emb.values))
        print(f"  embedded {start + len(chunk):>4}/{len(texts)}", flush=True)
    return np.asarray(vectors, dtype=np.float32)


def _zero_embeddings(n: int) -> np.ndarray:
    """Deterministic placeholder matrix used by --dry-run; pseudo-random so retrieval is sane."""
    settings = get_settings()
    rng = np.random.default_rng(seed=42)
    mat = rng.standard_normal((n, settings.embedding_dims), dtype=np.float32)
    return l2_normalize(mat)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SHL catalog index artifacts")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip embedding API calls; use deterministic placeholder vectors (for tests).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    settings = get_settings()
    started = time.perf_counter()
    print(f"Parsing catalog: {args.input}")
    items: list[CatalogItem] = normalize_catalog(args.input)
    print(f"  normalized {len(items)} items")

    print("Building BM25 index")
    bm25 = build_bm25(items, build_search_text)

    if args.dry_run:
        print("Computing placeholder embeddings (dry-run)")
        embeddings = _zero_embeddings(len(items))
    else:
        print(
            f"Computing dense embeddings via {settings.embedding_model} "
            f"@ {settings.embedding_dims} dims"
        )
        texts = [build_search_text(it) for it in items]
        embeddings = embed_corpus(texts)
        embeddings = l2_normalize(embeddings)

    print("Deriving category-coverage exemplars")
    coverage = derive_default_coverage(items)

    args.output.mkdir(parents=True, exist_ok=True)
    save_index(args.output, items, embeddings, bm25, coverage)

    meta = {
        "item_count": len(items),
        "embedding_model": settings.embedding_model if not args.dry_run else "PLACEHOLDER",
        "embedding_dims": settings.embedding_dims,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote artifacts to {args.output}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
