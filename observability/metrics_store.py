# -*- coding: utf-8 -*-
"""Aggregate lightweight metrics from stored query traces.

Metrics are computed from the SQLite trace store in a single query over compact
columns, instead of opening and JSON-parsing every trace file on each request.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from config.settings import PROJECT_ROOT, settings
from observability import trace_db


def _trace_dir() -> Path:
    output_dir = Path(settings.trace["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(value, 3)


def _avg(values: list[float]) -> float:
    return round(mean(values), 3) if values else 0.0


def _is_fallback(mode: str | None) -> bool:
    return bool(mode) and ("fallback" in mode or "failed" in mode)


def calculate_metrics() -> dict[str, Any]:
    rows = trace_db.fetch_metric_rows()
    total = len(rows)
    failed = sum(1 for row in rows if row.get("has_error"))
    success = total - failed

    total_latencies = [float(row.get("total_ms") or 0.0) for row in rows]
    retrieval_latencies = [float(row.get("retrieval_ms") or 0.0) for row in rows]
    generation_latencies = [float(row.get("generation_ms") or 0.0) for row in rows]
    # token/streaming metrics only exist for rows that actually generated
    ttfts = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms")]
    throughputs = [float(row["tokens_per_sec"]) for row in rows if row.get("tokens_per_sec")]
    completion_tokens = [int(row["completion_tokens"]) for row in rows if row.get("completion_tokens")]
    prompt_tokens = [int(row["prompt_tokens"]) for row in rows if row.get("prompt_tokens")]

    fallback_counts: Counter[str] = Counter()
    error_stage_counts: Counter[str] = Counter()
    error_code_counts: Counter[str] = Counter()

    for row in rows:
        for key in ("retrieval_mode", "rerank_mode", "generation_mode"):
            value = row.get(key)
            if _is_fallback(value):
                fallback_counts[value] += 1

        if row.get("has_error"):
            for stage in json.loads(row.get("error_stages") or "[]"):
                error_stage_counts[str(stage)] += 1
            for code in json.loads(row.get("error_codes") or "[]"):
                error_code_counts[str(code)] += 1

    return {
        "total_queries": total,
        "success_queries": success,
        "failed_queries": failed,
        "error_rate": round(failed / total, 4) if total else 0.0,
        "success_rate": round(success / total, 4) if total else 0.0,
        "avg_latency_ms": _avg(total_latencies),
        "p50_latency_ms": _percentile(total_latencies, 0.50),
        "p95_latency_ms": _percentile(total_latencies, 0.95),
        "fastest_latency_ms": round(min(total_latencies), 3) if total_latencies else 0.0,
        "slowest_latency_ms": round(max(total_latencies), 3) if total_latencies else 0.0,
        "avg_retrieval_ms": _avg(retrieval_latencies),
        "avg_generation_ms": _avg(generation_latencies),
        "avg_ttft_ms": _avg(ttfts),
        "p95_ttft_ms": _percentile(ttfts, 0.95),
        "avg_tokens_per_sec": _avg(throughputs),
        "avg_completion_tokens": _avg(completion_tokens),
        "avg_prompt_tokens": _avg(prompt_tokens),
        "fallback_counts": dict(fallback_counts),
        "error_stage_counts": dict(error_stage_counts),
        "error_code_counts": dict(error_code_counts),
        "trace_dir": str(_trace_dir()),
    }
