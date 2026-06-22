# -*- coding: utf-8 -*-
"""Ingestion trace persistence for document loading and indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT, settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ingestion_trace_dir() -> Path:
    output_dir = Path(settings.trace["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir / "ingestion"


@dataclass
class IngestionTrace:
    document: str
    collection_name: str
    document_id: str
    parse_method: str = "pypdf"
    ocr_used: bool = False
    quality_score: float = 0.0
    text_density: float = 0.0
    page_count: int = 0
    parent_chunks: int = 0
    child_chunks: int = 0
    embedding_status: str = "pending"
    milvus_insert_status: str = "pending"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "collection_name": self.collection_name,
            "document_id": self.document_id,
            "parse_method": self.parse_method,
            "ocr_used": self.ocr_used,
            "quality_score": self.quality_score,
            "text_density": self.text_density,
            "page_count": self.page_count,
            "parent_chunks": self.parent_chunks,
            "child_chunks": self.child_chunks,
            "embedding_status": self.embedding_status,
            "milvus_insert_status": self.milvus_insert_status,
            "warnings": self.warnings,
            "error": self.error,
            "error_code": self.error_code,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _trace_path(document_id: str) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in document_id)
    return ingestion_trace_dir() / f"{safe_id}.json"


def save_ingestion_trace(trace: IngestionTrace | dict[str, Any]) -> Path:
    data = trace.to_dict() if isinstance(trace, IngestionTrace) else dict(trace)
    data["updated_at"] = _now_iso()
    output_dir = ingestion_trace_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _trace_path(data.get("document_id") or data.get("document") or "document")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_ingestion_traces() -> list[dict[str, Any]]:
    output_dir = ingestion_trace_dir()
    if not output_dir.exists():
        return []
    traces = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            traces.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return traces


def delete_ingestion_trace(document_id: str) -> bool:
    path = _trace_path(document_id)
    if path.exists():
        path.unlink()
        return True
    return False


def update_ingestion_trace(document_id: str, **updates: Any) -> Path:
    path = _trace_path(document_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {
            "document": document_id,
            "document_id": document_id,
            "collection_name": settings.vector_db["collection"],
            "created_at": _now_iso(),
            "warnings": [],
        }
    data.update(updates)
    data["updated_at"] = _now_iso()
    return save_ingestion_trace(data)
