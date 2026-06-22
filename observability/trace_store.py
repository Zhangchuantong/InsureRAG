# -*- coding: utf-8 -*-
"""Persist RAG query traces.

Storage is backed by SQLite (see observability/trace_db.py). The public function
names are kept stable so existing callers (services, routes, dashboard) do not
need to change.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import settings
from observability import trace_db
from observability.trace_context import TraceContext


logger = logging.getLogger(__name__)


def save_trace(trace: TraceContext) -> str | None:
    if not settings.trace["enabled"]:
        logger.info("Trace disabled, skip saving trace_id=%s", trace.trace_id)
        return None

    payload = trace.to_dict()
    trace_id = trace_db.save_query_trace(payload)
    logger.info("Saved trace %s to %s", trace.trace_id, trace_db.db_path())
    return trace_id


def cleanup_old_traces() -> None:
    trace_db.cleanup_old_traces()


def load_trace(trace_id: str) -> dict[str, Any]:
    if not trace_id or any(char in trace_id for char in ("/", "\\", ":")):
        raise ValueError("Invalid trace_id")

    payload = trace_db.load_query_trace(trace_id)
    if payload is None:
        raise FileNotFoundError(f"Trace not found: {trace_id}")
    return payload
