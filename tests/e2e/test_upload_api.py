# -*- coding: utf-8 -*-
"""E2E tests for the upload endpoint + web UI wiring.

Ingestion itself (Milvus/BGE) is mocked so the HTTP contract can be tested
without the live model/vector stack.
"""

import shutil
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

import api.routes
from api.main import app
from config.settings import settings
from observability.ingestion_trace import load_ingestion_traces, update_ingestion_trace
from services.ingestion_service import reset_ingestion_trace


def test_index_page_is_served():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "InsureRAG" in resp.text
    assert "/api/upload" in resp.text  # frontend calls the upload endpoint


def test_upload_rejects_non_pdf():
    client = TestClient(app)
    resp = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 415


def test_upload_rejects_non_pdf_content_type_with_pdf_name():
    # explicit non-PDF content type is rejected even if the name ends with .pdf
    client = TestClient(app)
    resp = client.post("/api/upload", files={"file": ("a.pdf", b"x", "text/html")})
    assert resp.status_code == 415


def test_upload_rejects_empty_file():
    client = TestClient(app)
    resp = client.post("/api/upload", files={"file": ("a.pdf", b"", "application/pdf")})
    assert resp.status_code == 400


def test_upload_rejects_oversize_file(monkeypatch):
    monkeypatch.setattr(api.routes, "_max_upload_bytes", lambda: 8)
    client = TestClient(app)
    resp = client.post("/api/upload", files={"file": ("a.pdf", b"%PDF-1.4 oversized body", "application/pdf")})
    assert resp.status_code == 413


def test_upload_rejects_when_document_quota_reached(monkeypatch):
    monkeypatch.setattr(api.routes, "_uploaded_collections", lambda: {"upload_a", "upload_b"})
    monkeypatch.setitem(api.routes.settings.api, "max_uploaded_documents", 2)
    client = TestClient(app)
    resp = client.post("/api/upload", files={"file": ("brand_new.pdf", b"%PDF-1.4 small", "application/pdf")})
    assert resp.status_code == 400


def test_upload_accepts_pdf_and_schedules_ingestion(monkeypatch):
    def fake_save_upload(filename, data):
        return Path("ignored.pdf"), "policy_doc", "upload_abc123def456"

    scheduled = {"called": False}

    def fake_ingest(saved_path):
        scheduled["called"] = True

    monkeypatch.setattr(api.routes, "save_upload", fake_save_upload)
    monkeypatch.setattr(api.routes, "ingest_pdf_file", fake_ingest)
    # do not write a real ingestion trace from the test
    monkeypatch.setattr(api.routes, "reset_ingestion_trace", lambda *a, **k: None)

    client = TestClient(app)
    resp = client.post(
        "/api/upload",
        files={"file": ("保险条款.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == "policy_doc"
    assert body["collection"] == "upload_abc123def456"
    assert body["status"] == "processing"
    # background ingestion task ran (TestClient runs background tasks on response close)
    assert scheduled["called"] is True


def test_upload_status_unknown_for_missing_document(monkeypatch):
    monkeypatch.setattr(api.routes, "service_list_ingestion_traces", lambda: {"traces": []})
    client = TestClient(app)
    resp = client.get("/api/upload/nope/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "unknown"


def test_upload_status_ready_when_insert_succeeded(monkeypatch):
    trace = {
        "document_id": "policy_doc",
        "collection_name": "upload_abc123def456",
        "embedding_status": "success",
        "milvus_insert_status": "success",
        "parent_chunks": 6,
        "child_chunks": 40,
        "warnings": [],
    }
    monkeypatch.setattr(api.routes, "service_list_ingestion_traces", lambda: {"traces": [trace]})
    client = TestClient(app)
    resp = client.get("/api/upload/policy_doc/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["collection"] == "upload_abc123def456"
    assert data["child_chunks"] == 40


def test_list_uploads_returns_docs_with_status(monkeypatch):
    traces = [
        {
            "document_id": "doc_a", "document": "A.pdf", "collection_name": "upload_aaa",
            "milvus_insert_status": "success", "child_chunks": 40, "updated_at": "2026-06-20T10:00:00",
        },
        {
            "document_id": "doc_b", "document": "B.pdf", "collection_name": "upload_bbb",
            "error": "boom", "updated_at": "2026-06-20T11:00:00",
        },
        {
            "document_id": "保险条款", "document": "康宁终身保险条款", "collection_name": "insure_rag",
            "milvus_insert_status": "success", "child_chunks": 54, "updated_at": "2026-06-19T00:00:00",
        },
    ]
    monkeypatch.setattr(api.routes, "service_list_ingestion_traces", lambda: {"traces": traces})
    client = TestClient(app)
    resp = client.get("/api/uploads")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    by_id = {d["document_id"]: d for d in body["documents"]}
    assert by_id["doc_a"]["status"] == "ready"
    assert by_id["doc_a"]["collection"] == "upload_aaa"
    assert by_id["doc_a"]["deletable"] is True
    assert by_id["doc_b"]["status"] == "failed"
    # the built-in default collection must not be deletable from the UI
    assert by_id["保险条款"]["deletable"] is False
    # newest first
    assert body["documents"][0]["document_id"] == "doc_b"


def test_delete_upload_ok(monkeypatch):
    monkeypatch.setattr(
        api.routes,
        "delete_document",
        lambda doc_id: {"document_id": doc_id, "collection": "upload_aaa", "collection_dropped": True},
    )
    client = TestClient(app)
    resp = client.delete("/api/uploads/doc_a")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "deleted"
    assert body["collection_dropped"] is True


def test_delete_upload_not_found(monkeypatch):
    def _raise(doc_id):
        raise FileNotFoundError(doc_id)

    monkeypatch.setattr(api.routes, "delete_document", _raise)
    client = TestClient(app)
    resp = client.delete("/api/uploads/missing")
    assert resp.status_code == 404


def test_delete_upload_protects_builtin_collection(monkeypatch):
    def _raise(doc_id):
        raise PermissionError("built-in")

    monkeypatch.setattr(api.routes, "delete_document", _raise)
    client = TestClient(app)
    resp = client.delete("/api/uploads/保险条款")
    assert resp.status_code == 400


def test_index_page_has_multiupload_and_picker():
    client = TestClient(app)
    text = client.get("/").text
    assert "multiple" in text          # multi-select file input
    assert "/api/uploads" in text      # frontend calls the document list
    assert "docSelect" in text         # document picker present


def test_reset_clears_stale_failed_trace(monkeypatch):
    # isolate the trace dir in a project-local temp folder (avoids pytest tmp_path)
    base = Path(__file__).resolve().parents[2] / ".test_artifacts" / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(settings.trace, "output_dir", str(base))
    try:
        doc = "reset_doc"
        # a previous attempt left a failed trace
        update_ingestion_trace(doc, error="boom", error_code="OCR_ERROR", milvus_insert_status="failed")
        # a fresh upload resets it to a clean processing state
        reset_ingestion_trace(doc, "upload_reset", document="f.pdf")
        traces = {t["document_id"]: t for t in load_ingestion_traces()}
        assert traces[doc]["error"] is None
        assert traces[doc]["error_code"] is None
        assert traces[doc]["milvus_insert_status"] == "pending"
        assert traces[doc]["collection_name"] == "upload_reset"
    finally:
        shutil.rmtree(base, ignore_errors=True)
