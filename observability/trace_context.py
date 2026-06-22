# -*- coding: utf-8 -*-
"""Trace context for a single RAG query."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from observability.error_codes import classify_error


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class TraceContext:
    query: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    query_rewrite: str | None = None
    dense_results: list[dict[str, Any]] = field(default_factory=list)
    sparse_results: list[dict[str, Any]] = field(default_factory=list)
    rrf_results: list[dict[str, Any]] = field(default_factory=list)
    rerank_results: list[dict[str, Any]] = field(default_factory=list)
    final_context: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    latency: dict[str, float] = field(
        default_factory=lambda: {
            "embedding_ms": 0.0,
            "dense_search_ms": 0.0,
            "sparse_search_ms": 0.0,
            "rrf_ms": 0.0,
            "rerank_ms": 0.0,
            "ttft_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
        }
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    _started_at: float = field(default_factory=time.perf_counter, repr=False)

    def add_error(self, stage: str, error: BaseException | str, code: str | None = None) -> None:
        self.errors.append(
            {
                "stage": stage,
                "code": code or classify_error(stage, error),
                "message": str(error),
                "type": error.__class__.__name__ if isinstance(error, BaseException) else "Error",
            }
        )

    def add_warning(self, stage: str, message: str) -> None:
        self.warnings.append(
            {
                "stage": stage,
                "message": message,
            }
        )

    @contextmanager
    def timer(self, key: str) -> Iterator[None]:
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.latency[key] = round((time.perf_counter() - started_at) * 1000, 3)

    def finish(self) -> None:
        self.latency["total_ms"] = round((time.perf_counter() - self._started_at) * 1000, 3)

    def to_dict(self) -> dict[str, Any]:
        self.finish()
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "query_rewrite": self.query_rewrite,
            "dense_results": self.dense_results,
            "sparse_results": self.sparse_results,
            "rrf_results": self.rrf_results,
            "rerank_results": self.rerank_results,
            "final_context": self.final_context,
            "answer": self.answer,
            "latency": self.latency,
            "metadata": self.metadata,
            "errors": self.errors,
            "warnings": self.warnings,
            "created_at": self.created_at,
        }
