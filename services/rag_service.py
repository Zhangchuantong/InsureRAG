# -*- coding: utf-8 -*-
"""Shared RAG service used by FastAPI and MCP adapters."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from cache.cache_store import (
    answer_cache_key,
    cache_stats,
    evidence_overlap,
    get_json,
    search_cache_key,
    search_semantic_candidates,
    set_json,
    store_semantic_answer,
)
from config.settings import PROJECT_ROOT, settings
from observability.ingestion_trace import load_ingestion_traces
from observability.trace_context import TraceContext
from observability.trace_store import load_trace
from observability.trace_store import save_trace
from stage3_search import search
from stage4_generate import answer_with_hits_trace, answer_with_trace

logger = logging.getLogger(__name__)


def resolve_collection(collection: str | None = None) -> str:
    return collection or settings.vector_db["collection"]


def resolve_top_k(top_k: int | None = None) -> int:
    return int(top_k or settings.retrieval["final_top_k"])


def hit_to_evidence(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": hit.get("parent_text") or hit.get("content") or "",
        "collection_name": hit.get("collection_name"),
        "document_id": hit.get("document_id"),
        "document_type": hit.get("document_type"),
        "source": hit.get("source"),
        "page": hit.get("page"),
        "clause": hit.get("clause") or hit.get("clause_id"),
        "clause_id": hit.get("clause_id"),
        "parent_id": hit.get("parent_id"),
        "id": hit.get("id"),
        "score": hit.get("score"),
    }


def _evidence_to_hits(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for item in evidence:
        hits.append(
            {
                "score": item.get("score"),
                "collection_name": item.get("collection_name"),
                "document_id": item.get("document_id"),
                "document_type": item.get("document_type"),
                "source": item.get("source"),
                "page": item.get("page"),
                "clause_id": item.get("clause_id") or item.get("clause"),
                "parent_id": item.get("parent_id"),
                "id": item.get("id"),
                "child_text": item.get("content", ""),
                "parent_text": item.get("content", ""),
            }
        )
    return hits


def _load_trace_safe(trace_id: str) -> dict[str, Any]:
    try:
        return load_trace(trace_id)
    except Exception as exc:
        logger.warning("Failed to load trace %s: %s", trace_id, exc)
        return {}


def _cacheable_generation(trace: dict[str, Any]) -> tuple[str, bool]:
    metadata = trace.get("metadata", {})
    generation_mode = metadata.get("generation_mode", "")
    has_errors = bool(trace.get("errors"))
    return generation_mode, has_errors


def _store_answer_caches(question: str, result: dict[str, Any], trace: dict[str, Any]) -> None:
    generation_mode, has_errors = _cacheable_generation(trace)
    if generation_mode != "llm" or has_errors:
        return

    collection = result["collection"]
    top_k = resolve_top_k(result.get("top_k"))
    if settings.cache.get("answer_cache", True):
        set_json(
            answer_cache_key(question, collection, top_k),
            result,
            cache_type="answer",
        )
    store_semantic_answer(
        question,
        result,
        generation_mode=generation_mode,
        has_errors=has_errors,
    )


def _cached_answer_response(
    question: str,
    collection_name: str,
    cached: dict[str, Any],
    cache_type: str,
    started_at: float,
    **extra: Any,
) -> dict[str, Any]:
    # A cache hit is still a distinct request: write a lightweight trace for it so
    # that the returned trace_id refers to THIS request (not the historical one in
    # the cache) and so /api/metrics counts cache-hit traffic and latency.
    trace = TraceContext(query=question)
    trace._started_at = started_at  # total_ms reflects this request's cache latency
    trace.answer = cached.get("answer", "")
    trace.final_context = list(cached.get("evidence", []) or [])
    trace.metadata = {
        "collection": collection_name,
        "model": settings.llm["model"],
        "cache_hit": True,
        "cache_type": cache_type,
        "generation_mode": "cache_hit",
    }
    for key, value in extra.items():
        if value is not None:
            trace.metadata[key] = value
    save_trace(trace)

    response = dict(cached)
    response["cache_hit"] = True
    response["cache_type"] = cache_type
    response["trace_id"] = trace.trace_id
    response["latency_ms"] = trace.latency.get("total_ms") or round((time.perf_counter() - started_at) * 1000, 3)
    response.update(extra)
    return response


def query_insurance_clause(question: str, collection: str | None = None, top_k: int | None = None) -> dict[str, Any]:
    collection_name = resolve_collection(collection)
    final_top_k = resolve_top_k(top_k)
    started_at = time.perf_counter()

    exact_key = answer_cache_key(question, collection_name, final_top_k)
    if settings.cache.get("answer_cache", True):
        cached = get_json(exact_key)
        if cached:
            logger.info("Answer cache hit: type=exact key=%s", exact_key)
            return _cached_answer_response(question, collection_name, cached, "exact_answer", started_at)

    semantic_candidates = []
    min_semantic = float(settings.cache.get("semantic_retrieval_threshold", 0.93))
    if settings.cache.get("semantic_cache", True):
        semantic_candidates = search_semantic_candidates(
            question,
            collection=collection_name,
            top_k=final_top_k,
            min_similarity=min_semantic,
        )

    if semantic_candidates:
        best = semantic_candidates[0]
        similarity = float(best["similarity"])
        cached_answer = best["answer_json"]
        cached_evidence = best["evidence"]

        if similarity >= float(settings.cache.get("semantic_direct_threshold", 0.98)):
            logger.info("Semantic answer cache hit: type=direct similarity=%.4f", similarity)
            return _cached_answer_response(
                question,
                collection_name,
                cached_answer,
                "semantic_answer_direct",
                started_at,
                similarity=similarity,
            )

        if similarity >= float(settings.cache.get("semantic_answer_threshold", 0.95)):
            current_search = _run_search(question, collection_name, final_top_k)
            overlap, top1_match = evidence_overlap(current_search["evidence"], cached_evidence)
            require_top1 = bool(settings.cache.get("require_top1_evidence_match", True))
            if overlap >= float(settings.cache.get("evidence_overlap_threshold", 0.6)) and (top1_match or not require_top1):
                logger.info(
                    "Semantic answer cache hit: type=evidence_verified similarity=%.4f overlap=%.4f",
                    similarity,
                    overlap,
                )
                return _cached_answer_response(
                    question,
                    collection_name,
                    cached_answer,
                    "semantic_answer_evidence_verified",
                    started_at,
                    similarity=similarity,
                    evidence_overlap=overlap,
                    top1_evidence_match=top1_match,
                )

        if similarity >= min_semantic:
            logger.info("Semantic retrieval cache hit: similarity=%.4f", similarity)
            result = answer_with_hits_trace(
                question,
                _evidence_to_hits(cached_evidence),
                collection_name=collection_name,
                metadata={
                    "cache_hit": True,
                    "cache_type": "semantic_retrieval",
                    "similarity": similarity,
                },
            )
            trace = _load_trace_safe(result["trace_id"])
            response = _result_from_trace(
                result,
                trace,
                collection_name,
                final_top_k,
                started_at,
                cache_hit=True,
                cache_type="semantic_retrieval",
                similarity=similarity,
            )
            _store_answer_caches(question, response, trace)
            return response

    result = answer_with_trace(question, top_n=final_top_k, collection_name=collection_name)
    trace = _load_trace_safe(result["trace_id"])
    response = _result_from_trace(result, trace, collection_name, final_top_k, started_at)
    _store_answer_caches(question, response, trace)
    return response


def _result_from_trace(
    result: dict[str, Any],
    trace: dict[str, Any],
    collection_name: str,
    top_k: int,
    started_at: float,
    **extra: Any,
) -> dict[str, Any]:
    latency = trace.get("latency", {})
    metadata = trace.get("metadata", {})
    latency_ms = float(latency.get("total_ms") or round((time.perf_counter() - started_at) * 1000, 3))
    response = {
        "answer": result["answer"],
        "evidence": [hit_to_evidence(hit) for hit in trace.get("final_context", [])],
        "trace_id": result["trace_id"],
        "latency_ms": latency_ms,
        "collection": collection_name,
        "top_k": top_k,
        "cache_hit": False,
        "cache_type": "miss",
        "ttft_ms": latency.get("ttft_ms") or None,
        "tokens_per_sec": metadata.get("tokens_per_sec"),
        "prompt_tokens": metadata.get("prompt_tokens"),
        "completion_tokens": metadata.get("completion_tokens"),
    }
    response.update(extra)
    return response


def _run_search(
    query: str,
    collection_name: str,
    top_k: int,
    trace: TraceContext | None = None,
) -> dict[str, Any]:
    hits = search(query, top_n=top_k, collection_name=collection_name, trace=trace)
    return {
        "evidence": [hit_to_evidence(hit) for hit in hits],
        "collection": collection_name,
        "top_k": top_k,
    }


def search_related_clauses(query: str, top_k: int | None = None, collection: str | None = None) -> dict[str, Any]:
    collection_name = resolve_collection(collection)
    final_top_k = resolve_top_k(top_k)
    started_at = time.perf_counter()
    key = search_cache_key(query, collection_name, final_top_k)

    if settings.cache.get("search_cache", True):
        cached = get_json(key)
        if cached:
            # A search cache hit is still a distinct request: mint a lightweight
            # trace so the returned trace_id refers to THIS request and metrics
            # count search-cache traffic, mirroring the /api/query cache path.
            trace = TraceContext(query=query)
            trace._started_at = started_at
            trace.final_context = list(cached.get("evidence", []) or [])
            trace.metadata = {
                "final_k": final_top_k,
                "collection": collection_name,
                "model": settings.llm["model"],
                "request_type": "search_only",
                "cache_hit": True,
                "cache_type": "exact_search",
            }
            save_trace(trace)
            response = dict(cached)
            response["cache_hit"] = True
            response["cache_type"] = "exact_search"
            response["trace_id"] = trace.trace_id
            response["latency_ms"] = trace.latency.get("total_ms") or round((time.perf_counter() - started_at) * 1000, 3)
            return response

    trace = TraceContext(query=query)
    trace.metadata = {
        "final_k": final_top_k,
        "collection": collection_name,
        "model": settings.llm["model"],
        "request_type": "search_only",
    }
    result = _run_search(query, collection_name, final_top_k, trace=trace)
    save_trace(trace)
    result["trace_id"] = trace.trace_id
    result["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
    result["cache_hit"] = False
    result["cache_type"] = "miss"
    if settings.cache.get("search_cache", True):
        set_json(key, result, cache_type="search")
    return result


def _chunks_path() -> Path:
    return PROJECT_ROOT / "data" / "chunks.json"


def load_chunks() -> list[dict[str, Any]]:
    path = _chunks_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read chunks from %s: %s", path, exc)
        return []


def list_documents() -> dict[str, Any]:
    data_dir = PROJECT_ROOT / "data"
    documents = sorted(path.name for path in data_dir.glob("*.pdf"))
    chunks = load_chunks()
    if chunks:
        documents = sorted({chunk.get("source", "") for chunk in chunks if chunk.get("source")}) or documents

    by_doc: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        document_id = chunk.get("document_id") or chunk.get("source")
        if not document_id:
            continue
        item = by_doc.setdefault(
            str(document_id),
            {
                "document_id": str(document_id),
                "source": chunk.get("source"),
                "document_type": chunk.get("document_type"),
                "collection_name": chunk.get("collection_name", settings.vector_db["collection"]),
                "chunk_count": 0,
            },
        )
        item["chunk_count"] += 1

    return {
        "documents": documents,
        "document_items": sorted(by_doc.values(), key=lambda item: item["document_id"]),
        "chunk_count": len(chunks),
        "collection": settings.vector_db["collection"],
        "cache": cache_stats(),
    }


def list_ingestion_traces() -> dict[str, Any]:
    traces = load_ingestion_traces()
    return {
        "traces": traces,
        "count": len(traces),
        "collection": settings.vector_db["collection"],
    }


_CN_NUMBERS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _parse_chinese_number(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in _CN_NUMBERS:
        return _CN_NUMBERS[text]
    if "十" not in text:
        return None
    left, _, right = text.partition("十")
    tens = _CN_NUMBERS.get(left, 1 if left == "" else 0)
    ones = _CN_NUMBERS.get(right, 0 if right == "" else -1)
    if tens <= 0 or ones < 0:
        return None
    return tens * 10 + ones


def _clause_number(clause_id: str) -> int | None:
    normalized = clause_id.strip()
    match = re.search(r"第\s*([0-9一二两三四五六七八九十零]+)\s*条", normalized)
    if match:
        return _parse_chinese_number(match.group(1))
    return _parse_chinese_number(normalized)


def _unique_parent_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    parents = []
    for chunk in chunks:
        key = (
            chunk.get("collection_name"),
            chunk.get("document_id"),
            chunk.get("parent_id", chunk.get("id")),
        )
        if key in seen:
            continue
        seen.add(key)
        parents.append(chunk)
    return parents


def get_clause_detail(clause_id: str, collection: str | None = None) -> dict[str, Any]:
    collection_name = resolve_collection(collection)
    chunks = [
        chunk for chunk in load_chunks()
        if chunk.get("collection_name", settings.vector_db["collection"]) == collection_name
    ]
    parents = _unique_parent_chunks(chunks)
    target_number = _clause_number(clause_id)

    for chunk in parents:
        parent_id = chunk.get("parent_id")
        text = chunk.get("parent_text", "")
        current_clause_id = chunk.get("clause_id", "")
        if str(parent_id) == clause_id or str(chunk.get("id")) == clause_id:
            return _clause_response(clause_id, chunk)
        if current_clause_id == clause_id:
            return _clause_response(clause_id, chunk)
        if target_number is not None and current_clause_id:
            current_number = _clause_number(str(current_clause_id))
            if current_number == target_number:
                return _clause_response(clause_id, chunk)
        if clause_id and clause_id in text:
            return _clause_response(clause_id, chunk)

    return {
        "found": False,
        "clause_id": clause_id,
        "collection": collection_name,
        "message": "Clause not found in data/chunks.json.",
    }


def _clause_response(clause_id: str, chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "found": True,
        "clause_id": clause_id,
        "matched_clause_id": chunk.get("clause_id"),
        "collection": chunk.get("collection_name", settings.vector_db["collection"]),
        "document_id": chunk.get("document_id"),
        "document_type": chunk.get("document_type"),
        "parent_id": chunk.get("parent_id"),
        "source": chunk.get("source"),
        "page": chunk.get("page"),
        "content": chunk.get("parent_text", ""),
    }
