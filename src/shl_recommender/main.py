"""FastAPI application — entry point for /health and /chat."""

from __future__ import annotations

import os
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Load `.env` from the project root if present (no-op in production / Cloud Run).
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

from shl_recommender.agent.llm import LLMClient
from shl_recommender.agent.runner import Agent
from shl_recommender.catalog.loader import CatalogIndex, load_index
from shl_recommender.config import get_settings
from shl_recommender.observability.logging import configure_logging, get_logger
from shl_recommender.schemas import ChatRequest, ChatResponse, HealthResponse

configure_logging()
log = get_logger(__name__)


def _index_dir() -> Path:
    override = os.environ.get("SHL_INDEX_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "build"


class AppState:
    """Holds singletons loaded at app startup."""

    def __init__(self) -> None:
        self.ready: bool = False
        self.index: CatalogIndex | None = None
        self.agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("startup_begin")
    state = AppState()
    app.state.app_state = state

    index_dir = _index_dir()
    if index_dir.exists() and (index_dir / "catalog.parquet").exists():
        try:
            state.index = load_index(index_dir)
            llm = LLMClient()
            state.agent = Agent(index=state.index, llm=llm)
            state.ready = True
            log.info("index_loaded", path=str(index_dir), items=len(state.index.items))
        except Exception as exc:
            log.exception("index_load_failed", path=str(index_dir), error=str(exc))
    else:
        log.warning("index_artifacts_missing", path=str(index_dir))

    log.info("startup_complete", ready=state.ready)
    try:
        yield
    finally:
        log.info("shutdown")


app = FastAPI(
    title="SHL Assessment Recommender",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        log.exception("unhandled_error", error=str(exc))
        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal server error"},
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info("request_complete", status_code=response.status_code, latency_ms=elapsed_ms)
    response.headers["x-request-id"] = request_id
    structlog.contextvars.clear_contextvars()
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = _safe_validation_errors(exc.errors())
    log.warning("validation_error", errors=errors)
    return JSONResponse(status_code=422, content={"detail": errors})


def _safe_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop raw request `input` values so logs/responses do not echo message content."""
    safe: list[dict[str, Any]] = []
    for err in errors:
        cleaned = {k: v for k, v in err.items() if k not in {"input", "ctx"}}
        if "ctx" in err and "error" in err["ctx"]:
            cleaned["ctx"] = {"error": str(err["ctx"]["error"])}
        safe.append(cleaned)
    return safe


@app.get("/health", response_model=HealthResponse)
async def health(request: Request, response: Response) -> HealthResponse:
    """Readiness gate — 200 only when index is loaded."""
    state: AppState = request.app.state.app_state
    if not state.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, response_model_exclude_none=False)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Stateless conversational endpoint — full agent pipeline."""
    state: AppState = request.app.state.app_state
    if not state.ready or state.agent is None or state.index is None:
        log.warning("not_ready")
        return ChatResponse(
            reply="Service warming up — please retry in a moment.",
            recommendations=[],
            end_of_conversation=False,
        )

    log.info("chat_request", turn_index=len(payload.messages))
    try:
        result = await asyncio.wait_for(
            state.agent.chat(payload),
            timeout=get_settings().request_timeout_seconds,
        )
    except TimeoutError:
        log.warning("agent_timeout", timeout_seconds=get_settings().request_timeout_seconds)
        return ChatResponse(
            reply="I need a moment to process that. Please retry with the same conversation.",
            recommendations=[],
            end_of_conversation=False,
        )
    except Exception as exc:
        log.exception("agent_error", error=str(exc))
        return ChatResponse(
            reply="I hit an unexpected error. Could you rephrase or try again?",
            recommendations=[],
            end_of_conversation=False,
        )

    log.info(
        "chat_response",
        intent=result.decision.intent.value,
        is_final_turn=result.decision.is_final_turn,
        recommendations_count=len(result.response.recommendations),
        end_of_conversation=result.response.end_of_conversation,
        fallbacks_triggered=result.handler_result.fallbacks_triggered,
        validation_errors=result.handler_result.validation_errors,
        retrieval=result.handler_result.retrieval_stats,
        llm_calls=result.llm_stats,
        timings=result.timings,
    )
    return result.response
