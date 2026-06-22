# -*- coding: utf-8 -*-
"""SQLite-backed storage for query traces.

This replaces the previous "one JSON file per trace" design. Writing or reading
a single trace is now an O(1) indexed operation, and metrics aggregation is a
single SQL scan over compact columns instead of opening and JSON-parsing every
trace file on the hot path.

Only the Python standard library is used (sqlite3), so no new dependency is
required. WAL mode is enabled so concurrent readers do not block the writer.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()
_init_lock = threading.Lock()
_initialized = False
_save_counter = 0

# Run retention cleanup once every N writes to avoid an extra DELETE per query.
_CLEANUP_EVERY = 50

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_traces (
    trace_id        TEXT PRIMARY KEY,
    created_epoch   REAL NOT NULL,
    created_iso     TEXT,
    query           TEXT,
    total_ms        REAL,
    retrieval_ms    REAL,
    generation_ms   REAL,
    ttft_ms         REAL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    tokens_per_sec  REAL,
    has_error       INTEGER,
    retrieval_mode  TEXT,
    rerank_mode     TEXT,
    generation_mode TEXT,
    error_stages    TEXT,
    error_codes     TEXT,
    payload         TEXT
);
CREATE INDEX IF NOT EXISTS idx_query_traces_created ON query_traces(created_epoch);
"""

# Columns added after the initial release; ALTER existing DBs in-place on init.
_ADDED_COLUMNS = {
    "ttft_ms": "REAL",
    "prompt_tokens": "INTEGER",
    "completion_tokens": "INTEGER",
    "tokens_per_sec": "REAL",
}


def _trace_dir() -> Path:
    output_dir = Path(settings.trace["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def db_path() -> Path:
    return _trace_dir() / "traces.db"


def _connect() -> sqlite3.Connection:
    _trace_dir().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path(), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_init() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        with _connect() as conn:
            conn.executescript(_SCHEMA)
            _migrate_columns(conn)
            _backfill_legacy_json(conn)
        _initialized = True


def _migrate_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(query_traces)")}
    for column, decl in _ADDED_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE query_traces ADD COLUMN {column} {decl}")


def _iso_to_epoch(created_iso: str) -> float:
    if created_iso:
        try:
            return datetime.fromisoformat(created_iso).timestamp()
        except ValueError:
            pass
    return time.time()


def _derive_row(payload: dict[str, Any]) -> dict[str, Any]:
    latency = payload.get("latency", {}) or {}
    retrieval_ms = sum(
        float(latency.get(key) or 0.0)
        for key in ("embedding_ms", "dense_search_ms", "sparse_search_ms", "rrf_ms", "rerank_ms")
    )
    metadata = payload.get("metadata", {}) or {}
    errors = payload.get("errors", []) or []
    created_iso = str(payload.get("created_at") or "")
    return {
        "trace_id": payload.get("trace_id"),
        "created_epoch": _iso_to_epoch(created_iso),
        "created_iso": created_iso,
        "query": payload.get("query", ""),
        "total_ms": float(latency.get("total_ms") or 0.0),
        "retrieval_ms": round(retrieval_ms, 3),
        "generation_ms": float(latency.get("generation_ms") or 0.0),
        "ttft_ms": float(latency.get("ttft_ms")) if latency.get("ttft_ms") else None,
        "prompt_tokens": metadata.get("prompt_tokens"),
        "completion_tokens": metadata.get("completion_tokens"),
        "tokens_per_sec": metadata.get("tokens_per_sec"),
        "has_error": 1 if errors else 0,
        "retrieval_mode": metadata.get("retrieval_mode"),
        "rerank_mode": metadata.get("rerank_mode"),
        "generation_mode": metadata.get("generation_mode"),
        "error_stages": json.dumps([e.get("stage", "unknown") for e in errors], ensure_ascii=False),
        "error_codes": json.dumps([e.get("code", "UNKNOWN_ERROR") for e in errors], ensure_ascii=False),
        "payload": json.dumps(payload, ensure_ascii=False),
    }


_INSERT_SQL = """
INSERT OR REPLACE INTO query_traces (
    trace_id, created_epoch, created_iso, query, total_ms, retrieval_ms,
    generation_ms, ttft_ms, prompt_tokens, completion_tokens, tokens_per_sec,
    has_error, retrieval_mode, rerank_mode, generation_mode,
    error_stages, error_codes, payload
) VALUES (
    :trace_id, :created_epoch, :created_iso, :query, :total_ms, :retrieval_ms,
    :generation_ms, :ttft_ms, :prompt_tokens, :completion_tokens, :tokens_per_sec,
    :has_error, :retrieval_mode, :rerank_mode, :generation_mode,
    :error_stages, :error_codes, :payload
)
"""


def _backfill_legacy_json(conn: sqlite3.Connection) -> None:
    """One-time import of legacy per-trace JSON files into the database.

    Runs only when the table is empty. Files are copied, not deleted, so the
    migration is non-destructive; orphaned files can be removed separately.
    """
    existing = conn.execute("SELECT COUNT(*) FROM query_traces").fetchone()[0]
    if existing:
        return

    directory = _trace_dir()
    if not directory.exists():
        return

    imported = 0
    for path in directory.glob("*.json"):
        if not path.is_file() or path.name == "metrics.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload.setdefault("trace_id", path.stem)
        conn.execute(_INSERT_SQL, _derive_row(payload))
        imported += 1

    if imported:
        logger.info("Backfilled %s legacy trace JSON files into %s", imported, db_path())


def save_query_trace(payload: dict[str, Any]) -> str | None:
    _ensure_init()
    row = _derive_row(payload)
    if not row["trace_id"]:
        return None
    with _write_lock:
        with _connect() as conn:
            conn.execute(_INSERT_SQL, row)
    _maybe_cleanup()
    return str(row["trace_id"])


def load_query_trace(trace_id: str) -> dict[str, Any] | None:
    _ensure_init()
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT payload FROM query_traces WHERE trace_id = ?", (trace_id,)
        )
        result = cursor.fetchone()
    if not result:
        return None
    return json.loads(result[0])


def list_query_traces(limit: int = 50) -> list[dict[str, Any]]:
    _ensure_init()
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT trace_id, query, total_ms, created_iso "
            "FROM query_traces ORDER BY created_epoch DESC LIMIT ?",
            (int(limit),),
        )
        rows = cursor.fetchall()
    return [
        {"trace_id": row[0], "query": row[1], "total_ms": row[2], "created_at": row[3]}
        for row in rows
    ]


def count_query_traces() -> int:
    _ensure_init()
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM query_traces").fetchone()[0])


def fetch_metric_rows() -> list[dict[str, Any]]:
    """Return compact metric columns for every trace (no payload blob)."""
    _ensure_init()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT total_ms, retrieval_ms, generation_ms, ttft_ms, prompt_tokens, "
            "completion_tokens, tokens_per_sec, has_error, "
            "retrieval_mode, rerank_mode, generation_mode, error_stages, error_codes "
            "FROM query_traces"
        )
        return [dict(row) for row in cursor.fetchall()]


def _maybe_cleanup() -> None:
    global _save_counter
    _save_counter += 1
    if _save_counter % _CLEANUP_EVERY != 0:
        return
    cleanup_old_traces()


def cleanup_old_traces() -> None:
    _ensure_init()
    retention_days = int(settings.trace.get("retention_days", 7))
    max_rows = int(settings.trace.get("max_files", 1000))
    with _write_lock:
        with _connect() as conn:
            if retention_days > 0:
                cutoff = time.time() - retention_days * 24 * 60 * 60
                conn.execute("DELETE FROM query_traces WHERE created_epoch < ?", (cutoff,))
            if max_rows > 0:
                conn.execute(
                    "DELETE FROM query_traces WHERE trace_id IN ("
                    " SELECT trace_id FROM query_traces "
                    " ORDER BY created_epoch DESC LIMIT -1 OFFSET ?)",
                    (max_rows,),
                )
