# -*- coding: utf-8 -*-
"""Shared Milvus client helpers.

MilvusClient already manages low-level transport details; this module avoids
creating a new client object for every search request and serializes first-time
initialization under concurrent traffic.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, TypeVar

from pymilvus import MilvusClient

from config.settings import settings

logger = logging.getLogger(__name__)

_client: MilvusClient | None = None
_loaded_collections: set[str] = set()
_client_lock = threading.Lock()
_load_lock = threading.Lock()

T = TypeVar("T")

# Substrings that indicate a transport/connection problem (vs. a logical error
# like a missing field). Used to decide whether a reset + single retry is worth it.
_CONNECTION_ERROR_KEYWORDS = (
    "connect", "unavailable", "transport", "rpc", "deadline", "timeout",
    "timed out", "refused", "broken pipe", "channel", "socket",
    "reset by peer", "not ready", "no servers", "connection",
)


def get_milvus_client() -> MilvusClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = MilvusClient(uri=settings.milvus_uri)
    return _client


def reset_milvus_client() -> None:
    """Drop the cached client and loaded-collection set so the next call rebuilds
    a fresh connection. Used after a suspected connection failure (e.g. Milvus
    container restarted, invalidating the old client)."""
    global _client
    with _client_lock:
        _client = None
    with _load_lock:
        _loaded_collections.clear()


def is_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    text = f"{exc.__class__.__name__} {exc}".lower()
    return any(keyword in text for keyword in _CONNECTION_ERROR_KEYWORDS)


def run_with_milvus_retry(operation: Callable[[], T]) -> T:
    """Run a Milvus operation; on a connection-type failure, reset the client and
    retry exactly once. Non-connection errors and a second failure propagate."""
    try:
        return operation()
    except Exception as exc:
        if not is_connection_error(exc):
            raise
        logger.warning("Milvus client failed, resetting and retrying once: error=%s", exc.__class__.__name__)
        reset_milvus_client()
        return operation()


def load_collection_once(collection_name: str) -> None:
    if collection_name in _loaded_collections:
        return
    with _load_lock:
        if collection_name in _loaded_collections:
            return
        get_milvus_client().load_collection(collection_name)
        _loaded_collections.add(collection_name)


def forget_collection(collection_name: str) -> None:
    """Drop a collection from the 'already loaded' set.

    Call this whenever a collection is dropped or rebuilt, so the next search
    re-issues load_collection() against the new collection instead of assuming
    the stale (now-deleted) one is still loaded.
    """
    with _load_lock:
        _loaded_collections.discard(collection_name)
