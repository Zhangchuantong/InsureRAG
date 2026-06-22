# -*- coding: utf-8 -*-
"""Minimal HTTP routes that wrap the shared RAG service."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile

from api.auth import require_api_key
from chunking.parent_child_chunker import default_document_id
from config.settings import settings
from observability.error_codes import UNKNOWN_ERROR
from observability.health_checks import run_health_checks
from observability.request_logger import log_rag_request
from observability.trace_store import load_trace
from observability.metrics_store import calculate_metrics
from services.ingestion_service import (
    delete_document,
    ingest_pdf_file,
    mark_ingestion_failed,
    reset_ingestion_trace,
    save_upload,
    upload_collection_name,
)
from services.rag_service import (
    list_documents as service_list_documents,
    list_ingestion_traces as service_list_ingestion_traces,
    query_insurance_clause,
    search_related_clauses,
)

from api.schemas import (
    DocumentsResponse,
    EvidenceItem,
    HealthResponse,
    IngestionTracesResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    DeleteResponse,
    TraceResponse,
    UploadedDoc,
    UploadedDocsResponse,
    UploadResponse,
    UploadStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_evidence_item(hit: dict) -> EvidenceItem:
    return EvidenceItem(
        content=hit.get("content") or hit.get("parent_text") or "",
        collection_name=hit.get("collection_name"),
        document_id=hit.get("document_id"),
        document_type=hit.get("document_type"),
        source=hit.get("source"),
        page=hit.get("page"),
        clause=hit.get("clause"),
        clause_id=hit.get("clause_id"),
        parent_id=hit.get("parent_id"),
        id=hit.get("id"),
        score=hit.get("score"),
    )


@router.post("/api/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query_rag(payload: QueryRequest) -> QueryResponse:
    try:
        result = query_insurance_clause(
            payload.question,
            collection=payload.collection,
            top_k=payload.top_k,
        )
        trace = {}
        try:
            trace = load_trace(result["trace_id"])
        except Exception as exc:
            logger.warning("Failed to load trace for request log: %s", exc)
        log_rag_request(
            endpoint="/api/query",
            question=payload.question,
            trace_id=result.get("trace_id"),
            trace=trace,
            latency_ms=float(result.get("latency_ms") or 0.0),
            cache_hit=bool(result.get("cache_hit", False)),
            cache_type=result.get("cache_type", "miss"),
        )
        return QueryResponse(
            answer=result["answer"],
            evidence=[_to_evidence_item(hit) for hit in result["evidence"]],
            trace_id=result["trace_id"],
            latency_ms=result["latency_ms"],
            cache_hit=bool(result.get("cache_hit", False)),
            cache_type=result.get("cache_type", "miss"),
            similarity=result.get("similarity"),
            evidence_overlap=result.get("evidence_overlap"),
            top1_evidence_match=result.get("top1_evidence_match"),
            ttft_ms=result.get("ttft_ms"),
            tokens_per_sec=result.get("tokens_per_sec"),
            prompt_tokens=result.get("prompt_tokens"),
            completion_tokens=result.get("completion_tokens"),
        )
    except Exception as exc:
        logger.exception("Query API failed.")
        log_rag_request(
            endpoint="/api/query",
            question=payload.question,
            latency_ms=0.0,
            cache_hit=False,
            status="error",
            error_type=UNKNOWN_ERROR,
        )
        raise HTTPException(status_code=500, detail="RAG query failed. Please check server logs with the request trace.") from exc


@router.post("/api/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
def search_rag(payload: SearchRequest) -> SearchResponse:
    try:
        result = search_related_clauses(
            payload.question,
            top_k=payload.top_k,
            collection=payload.collection,
        )
        trace = {}
        trace_id = result.get("trace_id")
        if trace_id:
            try:
                trace = load_trace(trace_id)
            except Exception as exc:
                logger.warning("Failed to load search trace for request log: %s", exc)
        log_rag_request(
            endpoint="/api/search",
            question=payload.question,
            trace_id=trace_id,
            trace=trace,
            latency_ms=float(result.get("latency_ms") or 0.0),
            cache_hit=bool(result.get("cache_hit", False)),
            cache_type=result.get("cache_type", "miss"),
        )
        return SearchResponse(
            evidence=[_to_evidence_item(hit) for hit in result["evidence"]],
            latency_ms=result["latency_ms"],
            trace_id=result.get("trace_id"),
            cache_hit=bool(result.get("cache_hit", False)),
            cache_type=result.get("cache_type", "miss"),
        )
    except Exception as exc:
        logger.exception("Search API failed.")
        log_rag_request(
            endpoint="/api/search",
            question=payload.question,
            latency_ms=0.0,
            cache_hit=False,
            status="error",
            error_type=UNKNOWN_ERROR,
        )
        raise HTTPException(status_code=500, detail="Search failed. Please check server logs.") from exc


@router.get("/api/documents", response_model=DocumentsResponse, dependencies=[Depends(require_api_key)])
def list_documents() -> DocumentsResponse:
    result = service_list_documents()
    return DocumentsResponse(
        documents=result["documents"],
        document_items=result["document_items"],
        chunk_count=result["chunk_count"],
        collection=result["collection"],
    )


def _ingest_safely(saved_path) -> None:
    try:
        ingest_pdf_file(saved_path)
    except Exception as exc:
        logger.exception("Background ingestion failed for %s", saved_path)
        mark_ingestion_failed(saved_path, f"ingestion failed: {exc.__class__.__name__}")


def _derive_upload_status(trace: dict) -> str:
    if trace.get("error") or trace.get("error_code"):
        return "failed"
    if trace.get("milvus_insert_status") == "success":
        return "ready"
    if trace.get("milvus_insert_status") == "failed" or trace.get("embedding_status") == "failed":
        return "failed"
    return "processing"


# content types browsers commonly send for PDFs; empty/missing is tolerated and
# falls back to the filename-extension check below.
_ALLOWED_PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream"}


def _allowed_extensions() -> tuple[str, ...]:
    exts = settings.api.get("allowed_upload_extensions") or [".pdf"]
    return tuple(str(ext).lower() for ext in exts)


def _max_upload_bytes() -> int:
    return int(settings.api.get("max_upload_mb", 50)) * 1024 * 1024


def _uploaded_collections() -> set[str]:
    traces = service_list_ingestion_traces()["traces"]
    return {
        str(trace.get("collection_name"))
        for trace in traces
        if str(trace.get("collection_name", "")).startswith("upload_")
    }


@router.post("/api/upload", response_model=UploadResponse, dependencies=[Depends(require_api_key)])
async def upload_pdf(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> UploadResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(_allowed_extensions()):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")
    # Lenient content-type check: only reject when the browser explicitly sent a
    # non-PDF type; missing / octet-stream is allowed (the extension check stands).
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in _ALLOWED_PDF_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    max_bytes = _max_upload_bytes()
    over_limit = HTTPException(
        status_code=413,
        detail=f"File exceeds the {settings.api.get('max_upload_mb', 50)} MB upload limit.",
    )
    # Cheap early reject via Content-Length (with slack for multipart overhead).
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > max_bytes + 1024 * 1024:
        raise over_limit

    # Stream-read with a hard cap so an oversized file cannot blow up memory.
    data = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise over_limit
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Cap the number of distinct uploaded documents (re-uploading an existing one
    # reuses its collection and is always allowed).
    prospective_collection = upload_collection_name(default_document_id(filename))
    existing = _uploaded_collections()
    max_docs = int(settings.api.get("max_uploaded_documents", 20))
    if prospective_collection not in existing and len(existing) >= max_docs:
        raise HTTPException(
            status_code=400,
            detail=f"Reached the maximum of {max_docs} uploaded documents. Please delete some first.",
        )

    saved_path, document_id, collection = save_upload(filename, bytes(data))
    # Reset any stale trace from a previous attempt BEFORE returning, so the
    # client's first status poll sees "processing", not a leftover "failed".
    reset_ingestion_trace(document_id, collection, document=filename)
    # Ingestion (parse + embed + index) is slow; run it in the background and let
    # the client poll /api/upload/{document_id}/status.
    background_tasks.add_task(_ingest_safely, saved_path)
    logger.info("Upload accepted: document_id=%s collection=%s file=%s", document_id, collection, filename)
    return UploadResponse(document_id=document_id, collection=collection, filename=filename, status="processing")


@router.get(
    "/api/upload/{document_id}/status",
    response_model=UploadStatusResponse,
    dependencies=[Depends(require_api_key)],
)
def upload_status(document_id: str) -> UploadStatusResponse:
    traces = service_list_ingestion_traces()["traces"]
    match = next((t for t in traces if t.get("document_id") == document_id), None)
    if not match:
        return UploadStatusResponse(document_id=document_id, status="unknown")
    return UploadStatusResponse(
        document_id=document_id,
        collection=match.get("collection_name"),
        status=_derive_upload_status(match),
        embedding_status=match.get("embedding_status"),
        milvus_insert_status=match.get("milvus_insert_status"),
        parent_chunks=match.get("parent_chunks"),
        child_chunks=match.get("child_chunks"),
        error=match.get("error"),
        warnings=match.get("warnings", []),
    )


@router.get("/api/uploads", response_model=UploadedDocsResponse, dependencies=[Depends(require_api_key)])
def list_uploads() -> UploadedDocsResponse:
    """List every ingested document with its current status, so the UI can offer
    a picker of which document to query."""
    default_collection = settings.vector_db["collection"]
    traces = service_list_ingestion_traces()["traces"]
    docs = [
        UploadedDoc(
            document_id=str(trace.get("document_id", "")),
            document=str(trace.get("document") or trace.get("document_id", "")),
            collection=trace.get("collection_name"),
            status=_derive_upload_status(trace),
            child_chunks=trace.get("child_chunks"),
            updated_at=trace.get("updated_at"),
            deletable=bool(trace.get("collection_name")) and trace.get("collection_name") != default_collection,
        )
        for trace in traces
        if trace.get("document_id")
    ]
    docs.sort(key=lambda doc: doc.updated_at or "", reverse=True)
    return UploadedDocsResponse(documents=docs, count=len(docs))


@router.delete("/api/uploads/{document_id}", response_model=DeleteResponse, dependencies=[Depends(require_api_key)])
def delete_upload(document_id: str) -> DeleteResponse:
    try:
        result = delete_document(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail="Built-in collection cannot be deleted here.") from exc
    except Exception as exc:
        logger.exception("Delete document failed.")
        raise HTTPException(status_code=500, detail="Delete failed. Please check server logs.") from exc
    return DeleteResponse(**result)


@router.get("/api/ingestion-traces", response_model=IngestionTracesResponse, dependencies=[Depends(require_api_key)])
def get_ingestion_traces() -> IngestionTracesResponse:
    result = service_list_ingestion_traces()
    return IngestionTracesResponse(
        traces=result["traces"],
        count=result["count"],
        collection=result["collection"],
    )


@router.get("/api/metrics", response_model=MetricsResponse, dependencies=[Depends(require_api_key)])
def get_metrics() -> MetricsResponse:
    return MetricsResponse(**calculate_metrics())


@router.get("/api/traces/{trace_id}", response_model=TraceResponse, dependencies=[Depends(require_api_key)])
def get_trace(trace_id: str) -> TraceResponse:
    try:
        return TraceResponse(trace=load_trace(trace_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Trace not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid trace_id.") from exc
    except Exception as exc:
        logger.exception("Trace API failed.")
        raise HTTPException(status_code=500, detail="Read trace failed. Please check server logs.") from exc


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**run_health_checks())
