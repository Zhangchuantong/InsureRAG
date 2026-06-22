# -*- coding: utf-8 -*-
"""Exact and semantic cache for InsureRAG.

The public functions keep the original names used by services/rag_service.py.
Internally the cache prefers Redis when configured, with an optional JSON
fallback for local demos where Redis is not running.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT, settings
from observability.error_codes import CACHE_ERROR

logger = logging.getLogger(__name__)

_REDIS_CLIENT: Any | None = None
_REDIS_AVAILABLE: bool | None = None


def _configured_backend() -> str:
    return str(settings.cache.get("backend", "redis")).lower()


def _fallback_to_json() -> bool:
    return bool(settings.cache.get("fallback_to_json", True))


def _redis_prefix() -> str:
    return str(settings.cache.get("redis_prefix", "insurerag:cache")).strip(":")


def _redis_url() -> str:
    return str(settings.cache.get("redis_url", "redis://localhost:6379/0"))


def _redis_client() -> Any | None:
    global _REDIS_CLIENT, _REDIS_AVAILABLE

    if _REDIS_AVAILABLE is False:
        return None
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    try:
        import redis

        client = redis.Redis.from_url(_redis_url(), decode_responses=True)
        client.ping()
        _REDIS_CLIENT = client
        _REDIS_AVAILABLE = True
        return client
    except Exception as exc:
        _REDIS_AVAILABLE = False
        logger.warning("Redis cache unavailable: error_code=%s error=%s", CACHE_ERROR, exc.__class__.__name__)
        return None


def _handle_redis_failure(operation: str, exc: Exception) -> None:
    """A live Redis op failed mid-session (e.g. container restarted).

    Reset the cached client and mark Redis unavailable so subsequent calls
    gracefully fall back to the JSON store (or skip caching) instead of raising
    and turning every request into a 500.
    """
    global _REDIS_CLIENT, _REDIS_AVAILABLE
    logger.warning(
        "Redis op '%s' failed; resetting client and falling back: error_code=%s error=%s",
        operation,
        CACHE_ERROR,
        exc.__class__.__name__,
    )
    _REDIS_CLIENT = None
    _REDIS_AVAILABLE = False


def _use_redis() -> bool:
    return _configured_backend() == "redis" and _redis_client() is not None


def _use_json() -> bool:
    if _configured_backend() == "json":
        return True
    return _configured_backend() == "redis" and _fallback_to_json() and not _use_redis()


def cache_enabled() -> bool:
    if not bool(settings.cache.get("enabled", True)):
        return False
    return _configured_backend() in {"redis", "json"}


def _cache_path() -> Path:
    path = Path(settings.cache["path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _empty_store() -> dict[str, Any]:
    return {"kv_cache": {}, "semantic_answer_cache": []}


def _read_json_store() -> dict[str, Any]:
    path = _cache_path()
    if not path.exists():
        return _empty_store()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("kv_cache", {})
        data.setdefault("semantic_answer_cache", [])
        return data
    except Exception as exc:
        logger.warning("Failed to read cache file %s: error_code=%s error=%s", path, CACHE_ERROR, exc)
        return _empty_store()


def _write_json_store(data: dict[str, Any]) -> None:
    path = _cache_path()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _prune_expired(data: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    data["kv_cache"] = {
        key: value
        for key, value in data.get("kv_cache", {}).items()
        if float(value.get("expires_at", 0)) >= now
    }
    data["semantic_answer_cache"] = [
        value
        for value in data.get("semantic_answer_cache", [])
        if float(value.get("expires_at", 0)) >= now
    ]
    return data


def _redis_kv_key(key: str) -> str:
    return f"{_redis_prefix()}:kv:{key}"


def _redis_semantic_key() -> str:
    return f"{_redis_prefix()}:semantic_answer_cache"


def _redis_semantic_item_key(cache_id: str) -> str:
    return f"{_redis_prefix()}:semantic:{cache_id}"


def normalize_query(text: str) -> str:
    normalized = (text or "").strip().lower()
    replacements = {
        "？": "?",
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "！": "!",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return " ".join(normalized.split())


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def retrieval_config_hash() -> str:
    payload = {
        "dense_top_k": settings.retrieval["dense_top_k"],
        "sparse_top_k": settings.retrieval["sparse_top_k"],
        "final_top_k": settings.retrieval["final_top_k"],
        "rrf_k": settings.retrieval["rrf_k"],
        "reranker_enabled": settings.reranker["enabled"],
        "reranker_model": settings.reranker["model"],
        "embedding_model": settings.embedding["model_path"],
    }
    return _hash_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def answer_cache_key(question: str, collection: str, top_k: int) -> str:
    norm = normalize_query(question)
    return ":".join(
        [
            "answer",
            collection,
            settings.llm["model"],
            str(top_k),
            retrieval_config_hash(),
            _hash_text(norm),
        ]
    )


def search_cache_key(query: str, collection: str, top_k: int) -> str:
    norm = normalize_query(query)
    return ":".join(
        [
            "search",
            collection,
            str(top_k),
            retrieval_config_hash(),
            _hash_text(norm),
        ]
    )


def get_json(key: str) -> dict[str, Any] | None:
    if not cache_enabled():
        return None

    if _use_redis():
        try:
            client = _redis_client()
            if client is not None:
                raw = client.get(_redis_kv_key(key))
                if not raw:
                    return None
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    client.delete(_redis_kv_key(key))
                    return None
                item["hit_count"] = int(item.get("hit_count", 0)) + 1
                ttl = client.ttl(_redis_kv_key(key))
                if ttl and ttl > 0:
                    client.setex(_redis_kv_key(key), ttl, json.dumps(item, ensure_ascii=False))
                return item.get("value")
        except Exception as exc:
            _handle_redis_failure("get_json", exc)
            # fall through to JSON fallback below

    if _use_json():
        data = _prune_expired(_read_json_store())
        item = data["kv_cache"].get(key)
        if not item:
            _write_json_store(data)
            return None
        item["hit_count"] = int(item.get("hit_count", 0)) + 1
        _write_json_store(data)
        return item.get("value")

    return None


def set_json(key: str, value: dict[str, Any], cache_type: str, ttl_seconds: int | None = None) -> None:
    if not cache_enabled():
        return
    ttl = int(ttl_seconds or settings.cache["ttl_seconds"])
    now = time.time()

    if _use_redis():
        try:
            client = _redis_client()
            if client is not None:
                redis_key = _redis_kv_key(key)
                old_hit_count = 0
                old_created_at = now
                raw = client.get(redis_key)
                if raw:
                    try:
                        old = json.loads(raw)
                        old_hit_count = int(old.get("hit_count", 0))
                        old_created_at = float(old.get("created_at", now))
                    except json.JSONDecodeError:
                        pass
                item = {
                    "cache_type": cache_type,
                    "value": value,
                    "created_at": old_created_at,
                    "updated_at": now,
                    "hit_count": old_hit_count,
                }
                client.setex(redis_key, ttl, json.dumps(item, ensure_ascii=False))
                return
        except Exception as exc:
            _handle_redis_failure("set_json", exc)
            # fall through to JSON fallback below

    if _use_json():
        data = _prune_expired(_read_json_store())
        data["kv_cache"][key] = {
            "cache_type": cache_type,
            "value": value,
            "expires_at": now + ttl,
            "created_at": data["kv_cache"].get(key, {}).get("created_at", now),
            "updated_at": now,
            "hit_count": data["kv_cache"].get(key, {}).get("hit_count", 0),
        }
        _write_json_store(data)


def _query_embedding(question: str) -> list[float]:
    from stage3_search import get_bge

    bge = get_bge()
    embedding = bge([question])["dense"][0]
    return [float(value) for value in embedding]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def evidence_key(item: dict[str, Any]) -> str:
    return "::".join(
        str(item.get(key) or "")
        for key in ("collection_name", "document_id", "parent_id", "clause_id")
    )


def evidence_overlap(current: list[dict[str, Any]], cached: list[dict[str, Any]]) -> tuple[float, bool]:
    current_keys = [evidence_key(item) for item in current if evidence_key(item).strip(":")]
    cached_keys = [evidence_key(item) for item in cached if evidence_key(item).strip(":")]
    if not current_keys or not cached_keys:
        return 0.0, False
    set_current = set(current_keys)
    set_cached = set(cached_keys)
    overlap = len(set_current & set_cached) / len(set_current | set_cached)
    top1_match = current_keys[0] == cached_keys[0]
    return overlap, top1_match


def _semantic_item_matches(item: dict[str, Any], collection: str, top_k: int, config_hash: str) -> bool:
    if item.get("collection_name") != collection:
        return False
    if item.get("model") != settings.llm["model"]:
        return False
    if int(item.get("top_k", -1)) != int(top_k):
        return False
    if item.get("retrieval_config_hash") != config_hash:
        return False
    if item.get("generation_mode") != "llm" or item.get("has_errors"):
        return False
    return True


def _format_semantic_candidate(item: dict[str, Any], similarity: float) -> dict[str, Any]:
    return {
        "cache_id": item["cache_id"],
        "question": item["question"],
        "similarity": round(similarity, 6),
        "answer_json": item["answer"],
        "evidence": item["evidence"],
        "generation_mode": item["generation_mode"],
        "has_errors": bool(item["has_errors"]),
    }


def search_semantic_candidates(
    question: str,
    collection: str,
    top_k: int,
    min_similarity: float,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if not cache_enabled() or not bool(settings.cache.get("semantic_cache", True)):
        return []

    query_vector = _query_embedding(question)
    config_hash = retrieval_config_hash()
    candidates = []
    now = time.time()

    if _use_redis():
        try:
            client = _redis_client()
            if client is not None:
                expired_ids = []
                for cache_id in client.smembers(_redis_semantic_key()):
                    raw = client.get(_redis_semantic_item_key(cache_id))
                    if not raw:
                        expired_ids.append(cache_id)
                        continue
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        expired_ids.append(cache_id)
                        continue
                    if float(item.get("expires_at", 0)) < now:
                        expired_ids.append(cache_id)
                        continue
                    if not _semantic_item_matches(item, collection, top_k, config_hash):
                        continue
                    similarity = cosine_similarity(query_vector, item.get("embedding", []))
                    if similarity >= min_similarity:
                        candidates.append(_format_semantic_candidate(item, similarity))
                if expired_ids:
                    client.srem(_redis_semantic_key(), *expired_ids)
                candidates.sort(key=lambda candidate: candidate["similarity"], reverse=True)
                return candidates[:limit]
        except Exception as exc:
            _handle_redis_failure("search_semantic_candidates", exc)
            candidates = []
            # fall through to JSON fallback below

    if _use_json():
        data = _prune_expired(_read_json_store())
        for item in data.get("semantic_answer_cache", []):
            if not _semantic_item_matches(item, collection, top_k, config_hash):
                continue
            similarity = cosine_similarity(query_vector, item.get("embedding", []))
            if similarity >= min_similarity:
                candidates.append(_format_semantic_candidate(item, similarity))
        _write_json_store(data)
        candidates.sort(key=lambda candidate: candidate["similarity"], reverse=True)
        return candidates[:limit]

    return []


def store_semantic_answer(
    question: str,
    result: dict[str, Any],
    generation_mode: str,
    has_errors: bool,
    ttl_seconds: int | None = None,
) -> None:
    if not cache_enabled() or not bool(settings.cache.get("semantic_cache", True)):
        return
    if generation_mode != "llm" or has_errors:
        return

    collection = result.get("collection") or settings.vector_db["collection"]
    top_k = int(result.get("top_k") or settings.retrieval["final_top_k"])
    evidence = result.get("evidence") or []
    if not evidence:
        return

    ttl = int(ttl_seconds or settings.cache["ttl_seconds"])
    now = time.time()
    item = {
        "cache_id": uuid.uuid4().hex,
        "question": question,
        "question_norm": normalize_query(question),
        "embedding": _query_embedding(question),
        "answer": {
            "answer": result.get("answer", ""),
            "evidence": evidence,
            "trace_id": result.get("trace_id", ""),
            "latency_ms": result.get("latency_ms", 0.0),
            "collection": collection,
            "top_k": top_k,
            "model": settings.llm["model"],
            "generation_mode": generation_mode,
        },
        "evidence": evidence,
        "collection_name": collection,
        "model": settings.llm["model"],
        "top_k": top_k,
        "retrieval_config_hash": retrieval_config_hash(),
        "generation_mode": generation_mode,
        "has_errors": bool(has_errors),
        "expires_at": now + ttl,
        "created_at": now,
    }

    if _use_redis():
        try:
            client = _redis_client()
            if client is not None:
                client.setex(_redis_semantic_item_key(item["cache_id"]), ttl, json.dumps(item, ensure_ascii=False))
                client.sadd(_redis_semantic_key(), item["cache_id"])
                return
        except Exception as exc:
            _handle_redis_failure("store_semantic_answer", exc)
            # fall through to JSON fallback below

    if _use_json():
        data = _prune_expired(_read_json_store())
        data["semantic_answer_cache"].append(item)
        _write_json_store(data)


def _redis_cache_stats() -> dict[str, Any]:
    unavailable = {
        "enabled": bool(settings.cache.get("enabled", True)),
        "backend": "redis",
        "available": False,
        "redis_url": _redis_url(),
        "fallback_to_json": _fallback_to_json(),
    }
    client = _redis_client()
    if client is None:
        return unavailable
    try:
        kv_count = 0
        for _ in client.scan_iter(match=f"{_redis_prefix()}:kv:*", count=100):
            kv_count += 1
        semantic_ids = client.smembers(_redis_semantic_key())
        semantic_count = sum(1 for cache_id in semantic_ids if client.exists(_redis_semantic_item_key(cache_id)))
    except Exception as exc:
        _handle_redis_failure("cache_stats", exc)
        return unavailable
    return {
        "enabled": True,
        "backend": "redis",
        "available": True,
        "redis_url": _redis_url(),
        "redis_prefix": _redis_prefix(),
        "kv_cache_count": kv_count,
        "semantic_cache_count": semantic_count,
    }


def _json_cache_stats() -> dict[str, Any]:
    data = _prune_expired(_read_json_store())
    _write_json_store(data)
    return {
        "enabled": True,
        "backend": "json",
        "path": str(_cache_path()),
        "kv_cache_count": len(data.get("kv_cache", {})),
        "semantic_cache_count": len(data.get("semantic_answer_cache", [])),
    }


def cache_stats() -> dict[str, Any]:
    if not cache_enabled():
        return {"enabled": False}
    if _configured_backend() == "redis":
        stats = _redis_cache_stats()
        if stats.get("available") or not _fallback_to_json():
            return stats
        fallback = _json_cache_stats()
        fallback["configured_backend"] = "redis"
        fallback["redis_available"] = False
        return fallback
    return _json_cache_stats()
