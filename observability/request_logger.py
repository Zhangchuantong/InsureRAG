# -*- coding: utf-8 -*-
"""Request-level structured logging helpers."""

from __future__ import annotations

import json
import logging
from typing import Any

from observability.error_codes import first_error_code

logger = logging.getLogger("insurerag.request")


def log_rag_request(
    *,
    endpoint: str,
    question: str,
    latency_ms: float,
    cache_hit: bool,
    trace_id: str | None = None,
    trace: dict[str, Any] | None = None,
    cache_type: str | None = None,
    status: str = "ok",
    error_type: str | None = None,
) -> None:
    metadata = (trace or {}).get("metadata", {})
    errors = (trace or {}).get("errors", [])
    payload = {
        "event": "rag_request",
        "endpoint": endpoint,
        "status": status,
        "trace_id": trace_id or (trace or {}).get("trace_id"),
        "question": question,
        "latency_ms": round(float(latency_ms or 0.0), 3),
        "cache_hit": bool(cache_hit),
        "cache_type": cache_type or "miss",
        "retrieval_mode": metadata.get("retrieval_mode"),
        "rerank_mode": metadata.get("rerank_mode"),
        "generation_mode": metadata.get("generation_mode"),
        "error_type": error_type or first_error_code(errors),
    }
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))
