# -*- coding: utf-8 -*-
"""Stage 3: hybrid retrieval with Dense/Sparse search, RRF, and reranker."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any

from config.settings import settings
from observability.error_codes import EMBEDDING_ERROR, MILVUS_SEARCH_ERROR, RERANKER_ERROR
from observability.trace_context import TraceContext
from vectorstore.milvus_client import (
    get_milvus_client,
    is_connection_error,
    load_collection_once,
    run_with_milvus_retry,
)

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

VECTOR_DB_URI = settings.milvus_uri
DEFAULT_COLLECTION = settings.vector_db["collection"]

_bge = None
_reranker = None
_bge_lock = threading.Lock()
_reranker_lock = threading.Lock()


def configure_hf_offline() -> None:
    if bool(settings.embedding.get("local_files_only", True)) or bool(settings.reranker.get("local_files_only", True)):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


def get_bge():
    global _bge
    if _bge is None:
        with _bge_lock:
            if _bge is None:
                configure_hf_offline()
                from pymilvus.model.hybrid import BGEM3EmbeddingFunction

                _bge = BGEM3EmbeddingFunction(
                    model_name=settings.embedding["model_path"],
                    use_fp16=bool(settings.embedding["use_fp16"]),
                    device=settings.embedding["device"],
                    local_files_only=bool(settings.embedding.get("local_files_only", True)),
                )
    return _bge


def get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                configure_hf_offline()
                from sentence_transformers import CrossEncoder

                _reranker = CrossEncoder(
                    settings.reranker["model"],
                    device=settings.reranker["device"],
                    local_files_only=bool(settings.reranker.get("local_files_only", True)),
                )
    return _reranker


def bge_loaded() -> bool:
    """Cheap probe: is the embedding model already loaded? Does not trigger load."""
    return _bge is not None


def reranker_loaded() -> bool:
    """Cheap probe: is the reranker already loaded? Does not trigger load."""
    return _reranker is not None


def _hit_id(hit: Any) -> Any:
    entity = _hit_entity(hit)
    return hit.get("id") or hit.get("pk") or hit.get("primary_key") or entity.get("id")


def _hit_score(hit: Any) -> float:
    return float(hit.get("distance", hit.get("score", 0.0)))


def _hit_entity(hit: Any) -> dict[str, Any]:
    if not isinstance(hit, dict):
        return {}
    if "entity" in hit and isinstance(hit["entity"], dict):
        return dict(hit["entity"])
    return dict(hit)


def _serialize_hit(hit: Any, score: float | None = None) -> dict[str, Any]:
    entity = _hit_entity(hit)
    return {
        "id": _hit_id(hit),
        "score": _hit_score(hit) if score is None else float(score),
        "collection_name": entity.get("collection_name", ""),
        "document_id": entity.get("document_id", ""),
        "document_type": entity.get("document_type", ""),
        "source": entity.get("source", ""),
        "page": entity.get("page"),
        "clause_id": entity.get("clause_id", ""),
        "parent_id": entity.get("parent_id", ""),
        "child_text": entity.get("child_text", ""),
        "parent_text": entity.get("parent_text", ""),
    }


def _manual_rrf(dense_hits, sparse_hits, rrf_k: int, limit: int):
    fused: dict[Any, dict[str, Any]] = {}

    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            entity = _hit_entity(hit)
            key = _hit_id(hit)
            if key is None:
                key = (
                    entity.get("collection_name", ""),
                    entity.get("document_id", ""),
                    entity.get("source", ""),
                    entity.get("parent_id", ""),
                    entity.get("child_text", "")[:120],
                )
            if key not in fused:
                fused[key] = {"hit": hit, "score": 0.0}
            fused[key]["score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    return [(item["hit"], item["score"]) for item in ranked[:limit]]


def _parent_dedup_key(hit: Any) -> tuple[Any, ...]:
    entity = _hit_entity(hit)
    parent_id = entity.get("parent_id")
    if parent_id:
        return (
            entity.get("collection_name", ""),
            entity.get("document_id", ""),
            parent_id,
        )
    return (
        entity.get("collection_name", ""),
        entity.get("document_id", ""),
        entity.get("source", ""),
        entity.get("clause_id", ""),
        entity.get("parent_text", "")[:200],
    )


def _dedup_ranked_by_parent(ranked, top_n: int):
    seen: set[tuple[Any, ...]] = set()
    deduped = []
    duplicate_count = 0

    for candidate, score in ranked:
        key = _parent_dedup_key(candidate)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append((candidate, score))
        if len(deduped) >= top_n:
            break

    return deduped, duplicate_count


def _output_fields() -> list[str]:
    return [
        "collection_name",
        "document_id",
        "document_type",
        "source",
        "page",
        "clause_id",
        "parent_id",
        "child_text",
        "parent_text",
    ]


def hybrid_retrieve(
    client,
    query: str,
    trace: TraceContext | None = None,
    collection_name: str | None = None,
):
    collection = collection_name or DEFAULT_COLLECTION
    dense_top_k = int(settings.retrieval["dense_top_k"])
    sparse_top_k = int(settings.retrieval["sparse_top_k"])
    rrf_k = int(settings.retrieval["rrf_k"])
    hybrid_limit = max(dense_top_k, sparse_top_k)

    bge = get_bge()
    try:
        started_at = time.perf_counter()
        q_emb = bge([query])
        q_dense = q_emb["dense"][0]
        q_sparse = q_emb["sparse"][[0]]
        if trace:
            trace.latency["embedding_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
    except Exception as exc:
        logger.exception("Embedding failed, retrieval cannot continue.")
        if trace:
            trace.add_error("embedding", exc, code=EMBEDDING_ERROR)
            trace.metadata["retrieval_mode"] = "embedding_failed"
        return []

    dense_hits = []
    sparse_hits = []

    try:
        started_at = time.perf_counter()
        dense_results = client.search(
            collection_name=collection,
            data=[q_dense],
            anns_field="dense_vector",
            search_params={"metric_type": "IP"},
            limit=dense_top_k,
            output_fields=_output_fields(),
        )
        dense_hits = dense_results[0]
        if trace:
            trace.latency["dense_search_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
            trace.dense_results = [_serialize_hit(hit) for hit in dense_hits]
    except Exception as exc:
        if is_connection_error(exc):
            raise  # let the caller reset the client and retry once
        logger.warning("Dense search failed, will fallback to sparse-only if sparse succeeds: %s", exc)
        if trace:
            trace.add_error("dense_search", exc, code=MILVUS_SEARCH_ERROR)

    try:
        started_at = time.perf_counter()
        sparse_results = client.search(
            collection_name=collection,
            data=[q_sparse],
            anns_field="sparse_vector",
            search_params={"metric_type": "IP"},
            limit=sparse_top_k,
            output_fields=_output_fields(),
        )
        sparse_hits = sparse_results[0]
        if trace:
            trace.latency["sparse_search_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
            trace.sparse_results = [_serialize_hit(hit) for hit in sparse_hits]
    except Exception as exc:
        if is_connection_error(exc):
            raise  # let the caller reset the client and retry once
        logger.warning("Sparse search failed, will fallback to dense-only if dense succeeds: %s", exc)
        if trace:
            trace.add_error("sparse_search", exc, code=MILVUS_SEARCH_ERROR)

    try:
        started_at = time.perf_counter()
        if trace:
            if dense_hits and sparse_hits:
                trace.metadata["retrieval_mode"] = "hybrid_rrf"
            elif dense_hits:
                trace.metadata["retrieval_mode"] = "dense_only_fallback"
                trace.add_warning("retrieval", "Sparse search failed or returned no results; using dense-only fallback.")
            elif sparse_hits:
                trace.metadata["retrieval_mode"] = "sparse_only_fallback"
                trace.add_warning("retrieval", "Dense search failed or returned no results; using sparse-only fallback.")
            else:
                trace.metadata["retrieval_mode"] = "no_results"
        rrf_ranked = _manual_rrf(dense_hits, sparse_hits, rrf_k=rrf_k, limit=hybrid_limit)
        if trace:
            trace.latency["rrf_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
            trace.rrf_results = [_serialize_hit(hit, score=score) for hit, score in rrf_ranked]
        return [hit for hit, _ in rrf_ranked]
    except Exception as exc:
        logger.warning("RRF fusion failed, fallback to available raw retrieval hits: %s", exc)
        if trace:
            trace.add_error("rrf", exc, code=MILVUS_SEARCH_ERROR)
            trace.metadata["retrieval_mode"] = "rrf_failed_raw_fallback"
        return dense_hits or sparse_hits


def rerank(query: str, candidates, top_n: int, trace: TraceContext | None = None):
    if not settings.reranker["enabled"]:
        ranked = [(cand, _hit_score(cand)) for cand in candidates[:top_n]]
        if trace:
            trace.metadata["rerank_mode"] = "disabled_rrf_topk"
            trace.rerank_results = [_serialize_hit(cand, score=score) for cand, score in ranked]
        return ranked

    try:
        started_at = time.perf_counter()
        reranker = get_reranker()
        pairs = [[query, _hit_entity(candidate).get("child_text", "")] for candidate in candidates]
        scores = [float(score) for score in reranker.predict(pairs)]
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)[:top_n]
        if trace:
            trace.latency["rerank_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
            trace.metadata["rerank_mode"] = "cross_encoder"
            trace.rerank_results = [_serialize_hit(cand, score=score) for cand, score in ranked]
        return ranked
    except Exception as exc:
        logger.warning("Reranker failed, fallback to RRF TopK: %s", exc)
        if trace:
            trace.add_error("rerank", exc, code=RERANKER_ERROR)
            trace.add_warning("rerank", "Reranker failed; using RRF TopK fallback.")
            trace.metadata["rerank_mode"] = "rrf_topk_fallback"
        ranked = []
        for index, cand in enumerate(candidates[:top_n]):
            if trace and index < len(trace.rrf_results):
                score = float(trace.rrf_results[index]["score"])
            else:
                score = _hit_score(cand)
            ranked.append((cand, score))
        if trace:
            trace.rerank_results = [_serialize_hit(cand, score=score) for cand, score in ranked]
        return ranked


def search(
    query: str,
    top_n: int | None = None,
    trace: TraceContext | None = None,
    collection_name: str | None = None,
):
    top_n = top_n or int(settings.retrieval["final_top_k"])
    collection = collection_name or DEFAULT_COLLECTION

    def _retrieve():
        client = get_milvus_client()
        load_collection_once(collection)
        return hybrid_retrieve(client, query, trace=trace, collection_name=collection)

    try:
        # On a Milvus connection failure, reset the client and retry once; if it
        # still fails, fall back to empty candidates (no crash) and record the error.
        candidates = run_with_milvus_retry(_retrieve)
    except Exception as exc:
        logger.warning("Milvus retrieval failed after retry: %s", exc)
        if trace:
            trace.add_error("milvus", exc, code=MILVUS_SEARCH_ERROR)
            trace.metadata["retrieval_mode"] = "milvus_failed"
        candidates = []

    rerank_limit = min(len(candidates), max(top_n * 3, top_n))
    ranked = rerank(query, candidates, top_n=rerank_limit, trace=trace)
    ranked, duplicate_count = _dedup_ranked_by_parent(ranked, top_n=top_n)

    out = []
    for candidate, score in ranked:
        item = _serialize_hit(candidate, score=score)
        out.append(item)
    if trace:
        trace.metadata["parent_dedup_enabled"] = True
        trace.metadata["parent_dedup_removed"] = duplicate_count
        trace.metadata["final_top_k"] = top_n
        trace.final_context = out
    return out


if __name__ == "__main__":
    for question in ["等待期内生病了能赔吗？", "酒后驾驶发生事故赔吗？"]:
        print("\n" + "=" * 50)
        print("问题：", question)
        for result in search(question, top_n=2):
            print(f"  [score={result['score']:.3f}] {result['child_text'][:50]}...")
