# -*- coding: utf-8 -*-
"""Dependency health checks for the FastAPI /health endpoint."""

from __future__ import annotations

import time
from typing import Any, Callable

from openai import OpenAI

from cache.cache_store import cache_stats
from config.settings import settings
from observability.error_codes import CACHE_ERROR, LLM_TIMEOUT, MILVUS_SEARCH_ERROR, RERANKER_ERROR
from vectorstore.milvus_client import get_milvus_client


def _check(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        result = fn()
        result.setdefault("status", "ok")
        result["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
        return result
    except Exception as exc:
        return {
            "status": "error",
            "name": name,
            "error": exc.__class__.__name__,
            "message": str(exc),
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
        }


def check_redis() -> dict[str, Any]:
    stats = cache_stats()
    if not settings.cache.get("enabled", True):
        return {"status": "disabled", "backend": "disabled"}
    if settings.cache.get("backend") == "redis" and not stats.get("available"):
        return {
            "status": "error",
            "error_code": CACHE_ERROR,
            **stats,
        }
    return {"status": "ok", **stats}


def check_milvus() -> dict[str, Any]:
    client = get_milvus_client()
    collections = client.list_collections()
    collection = settings.vector_db["collection"]
    return {
        "status": "ok" if collection in collections else "degraded",
        "uri": settings.milvus_uri,
        "collection": collection,
        "collection_exists": collection in collections,
        "collections": collections,
        "error_code": None if collection in collections else MILVUS_SEARCH_ERROR,
    }


def check_vllm() -> dict[str, Any]:
    client = OpenAI(
        api_key=settings.llm["api_key"],
        base_url=settings.llm["base_url"],
        timeout=float(settings.llm.get("health_timeout_seconds", 3)),
    )
    models = client.models.list()
    model_ids = [item.id for item in models.data]
    configured_model = settings.llm["model"]
    return {
        "status": "ok" if configured_model in model_ids or model_ids else "degraded",
        "base_url": settings.llm["base_url"],
        "model": configured_model,
        "available_models": model_ids[:10],
        "error_code": None if model_ids else LLM_TIMEOUT,
    }


def check_embedding() -> dict[str, Any]:
    # Health checks must stay cheap: report whether the model is already loaded
    # instead of triggering a (potentially multi-second / OOM-prone) load here.
    # The model is loaded lazily on the first real query.
    from stage3_search import bge_loaded

    loaded = bge_loaded()
    return {
        "status": "ok",
        "loaded": loaded,
        "note": None if loaded else "Model not loaded yet; loads lazily on first query.",
        "model": settings.embedding["model_path"],
        "device": settings.embedding["device"],
    }


def check_reranker() -> dict[str, Any]:
    if not settings.reranker.get("enabled", True):
        return {
            "status": "disabled",
            "model": settings.reranker["model"],
        }

    from stage3_search import reranker_loaded

    loaded = reranker_loaded()
    return {
        "status": "ok",
        "loaded": loaded,
        "note": None if loaded else "Model not loaded yet; loads lazily on first query.",
        "model": settings.reranker["model"],
        "device": settings.reranker["device"],
    }


def run_health_checks() -> dict[str, Any]:
    checks = {
        "redis": _check("redis", check_redis),
        "milvus": _check("milvus", check_milvus),
        "vllm": _check("vllm", check_vllm),
        "embedding": _check("embedding", check_embedding),
        "reranker": _check("reranker", check_reranker),
    }

    if checks["reranker"]["status"] == "error":
        checks["reranker"]["error_code"] = RERANKER_ERROR
    if checks["vllm"]["status"] == "error":
        checks["vllm"]["error_code"] = LLM_TIMEOUT
    if checks["milvus"]["status"] == "error":
        checks["milvus"]["error_code"] = MILVUS_SEARCH_ERROR
    if checks["redis"]["status"] == "error":
        checks["redis"]["error_code"] = CACHE_ERROR

    statuses = {item["status"] for item in checks.values()}
    overall = "ok"
    if "error" in statuses:
        overall = "error"
    elif "degraded" in statuses:
        overall = "degraded"

    return {
        "status": overall,
        "collection": settings.vector_db["collection"],
        "llm_model": settings.llm["model"],
        "milvus_uri": settings.milvus_uri,
        "checks": checks,
    }
