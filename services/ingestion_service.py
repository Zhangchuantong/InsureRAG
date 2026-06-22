# -*- coding: utf-8 -*-
"""Ingest a single uploaded PDF into its own Milvus collection.

Each uploaded document goes into a dedicated collection named ``upload_<hash>``
so that question answering can be scoped to exactly that PDF via the existing
``collection`` parameter — no changes to retrieval filters or cache keys needed.
Re-uploading the same file reuses the same collection (drop + rebuild), so it
never disturbs other uploaded documents.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

from chunking.parent_child_chunker import build_chunks, default_document_id
from config.settings import PROJECT_ROOT, settings
from observability.error_codes import OCR_ERROR
from observability.ingestion_trace import (
    IngestionTrace,
    delete_ingestion_trace,
    load_ingestion_traces,
    save_ingestion_trace,
    update_ingestion_trace,
)
from stage1_load_split import load_pdfs
from stage2_build_db import index_chunks
from vectorstore.milvus_client import forget_collection, get_milvus_client

logger = logging.getLogger(__name__)

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


def upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def upload_collection_name(document_id: str) -> str:
    """ASCII-safe, stable Milvus collection name for an uploaded document.

    Milvus collection names must match ^[A-Za-z_][A-Za-z0-9_]*$, so we hash the
    (possibly non-ASCII) document_id instead of using it directly.
    """
    digest = hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:12]
    return f"upload_{digest}"


def save_upload(filename: str, data: bytes) -> tuple[Path, str, str]:
    """Persist raw upload bytes into a per-document subdirectory.

    Each document gets its own folder data/uploads/<document_id>/ containing just
    its PDF, so stage1's directory-scanning load_pdfs() processes exactly this one
    file — no system temp dir / copy needed (which on Windows could not be cleaned
    up while PyMuPDF held the file open). Returns (saved_path, document_id, collection).
    """
    document_id = default_document_id(filename)
    doc_dir = upload_dir() / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    saved_path = doc_dir / f"{document_id}.pdf"
    saved_path.write_bytes(data)
    return saved_path, document_id, upload_collection_name(document_id)


def reset_ingestion_trace(document_id: str, collection: str, document: str | None = None) -> None:
    """Overwrite any prior ingestion trace for this document with a fresh
    'processing' (pending) state.

    Without this, re-uploading a file that previously failed would leave the old
    'error'/'failed' trace in place until OCR/embedding finishes (~minute later),
    and the client's first status poll would read that stale failure and give up.
    """
    trace = IngestionTrace(
        document=document or document_id,
        collection_name=collection,
        document_id=document_id,
    )
    save_ingestion_trace(trace)


def delete_document(document_id: str) -> dict[str, Any]:
    """Remove an uploaded document: drop its Milvus collection, delete its
    ingestion trace and the saved PDF folder.

    Raises FileNotFoundError if the document is unknown, and PermissionError if
    it maps to the built-in default collection (which is managed by the CLI and
    must not be dropped through this endpoint).
    """
    traces = {t.get("document_id"): t for t in load_ingestion_traces()}
    trace = traces.get(document_id)
    if not trace:
        raise FileNotFoundError(document_id)

    collection = str(trace.get("collection_name") or "")
    if collection and collection == settings.vector_db["collection"]:
        raise PermissionError("built-in default collection cannot be deleted here")

    dropped = False
    if collection:
        try:
            client = get_milvus_client()
            if client.has_collection(collection):
                client.drop_collection(collection)
                dropped = True
            forget_collection(collection)
        except Exception:
            logger.exception("Failed to drop collection %s", collection)

    delete_ingestion_trace(document_id)
    doc_dir = upload_dir() / document_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir, ignore_errors=True)

    logger.info("Deleted document_id=%s collection=%s dropped=%s", document_id, collection, dropped)
    return {"document_id": document_id, "collection": collection or None, "collection_dropped": dropped}


def mark_ingestion_failed(saved_path: Path, message: str) -> None:
    """Record an unexpected ingestion failure on the trace so polling clients see
    'failed' instead of an endless 'processing'."""
    document_id = default_document_id(Path(saved_path).name)
    try:
        # also re-assert the upload collection so the failed doc stays deletable
        # (load_pdfs may have reset collection_name to the default during parsing).
        update_ingestion_trace(
            document_id,
            error=message,
            error_code=OCR_ERROR,
            collection_name=upload_collection_name(document_id),
        )
    except Exception:
        logger.exception("Failed to mark ingestion trace failed for %s", document_id)


def ingest_pdf_file(saved_path: Path) -> dict[str, Any]:
    """Parse, chunk, embed and index one uploaded PDF into its own collection.

    Designed to run in a background task. Progress/outcome is recorded in the
    ingestion trace (queryable via /api/upload/{document_id}/status), so failures
    here are reflected there rather than crashing the request.
    """
    saved_path = Path(saved_path)
    document_id = default_document_id(saved_path.name)
    collection = upload_collection_name(document_id)

    # The uploaded file lives alone in its own folder, so scan that folder.
    docs = load_pdfs(saved_path.parent)

    if not docs:
        logger.warning("Upload ingestion produced no docs for %s", document_id)
        update_ingestion_trace(document_id, collection_name=collection)
        return {
            "document_id": document_id,
            "collection": collection,
            "status": "failed",
            "message": "PDF parsing / quality check failed. See ingestion trace.",
        }

    chunks = build_chunks(docs, collection_name=collection)
    parent_ids = {chunk.get("parent_id") for chunk in chunks}
    update_ingestion_trace(
        document_id,
        collection_name=collection,
        parent_chunks=len(parent_ids),
        child_chunks=len(chunks),
    )
    result = index_chunks(chunks, collection_name=collection)
    logger.info("Upload ingested: document_id=%s collection=%s chunks=%s",
                document_id, collection, result.get("chunk_count"))
    return {
        "document_id": document_id,
        "collection": collection,
        "status": "success",
        "chunk_count": result.get("chunk_count", len(chunks)),
    }
