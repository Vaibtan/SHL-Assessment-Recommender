# Purpose: Structured JSON logging — stdout sink, Cloud Logging-compatible.

from __future__ import annotations

import logging
import os
import sys

import structlog
from structlog.types import EventDict, Processor


def _add_severity(logger: object, name: str, event_dict: EventDict) -> EventDict:
    """Map structlog level → Cloud Logging `severity` field."""
    level = event_dict.pop("level", name).upper()
    cloud_level = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }.get(level, "DEFAULT")
    event_dict["severity"] = cloud_level
    return event_dict


def configure_logging(level: str | None = None) -> None:
    """Configure structlog + stdlib logging once at process start."""
    log_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _add_severity,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "shl_recommender") -> structlog.stdlib.BoundLogger:
    """Get a bound logger for the given name."""
    return structlog.get_logger(name)
