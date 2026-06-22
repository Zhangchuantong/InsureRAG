# -*- coding: utf-8 -*-
"""Pydantic schemas for the minimal InsureRAG HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from config.settings import settings


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    collection: str = Field(
        default_factory=lambda: settings.vector_db["collection"],
        description="当前最小服务仅支持配置文件中的 collection",
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="最终返回上下文数量")


class EvidenceItem(BaseModel):
    content: str
    collection_name: str | None = None
    document_id: str | None = None
    document_type: str | None = None
    source: str | None = None
    page: int | None = None
    clause: str | None = None
    clause_id: str | None = None
    parent_id: str | None = None
    id: int | str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem]
    trace_id: str
    latency_ms: float
    cache_hit: bool = False
    cache_type: str = "miss"
    similarity: float | None = None
    evidence_overlap: float | None = None
    top1_evidence_match: bool | None = None
    ttft_ms: float | None = None
    tokens_per_sec: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, description="检索问题")
    collection: str = Field(
        default_factory=lambda: settings.vector_db["collection"],
        description="当前最小服务仅支持配置文件中的 collection",
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="检索返回数量")


class SearchResponse(BaseModel):
    evidence: list[EvidenceItem]
    latency_ms: float
    trace_id: str | None = None
    cache_hit: bool = False
    cache_type: str = "miss"


class DocumentsResponse(BaseModel):
    documents: list[str]
    document_items: list[dict[str, Any]] = []
    chunk_count: int
    collection: str


class HealthResponse(BaseModel):
    status: str
    collection: str
    llm_model: str
    milvus_uri: str
    checks: dict[str, Any] = {}


class UploadResponse(BaseModel):
    document_id: str
    collection: str
    filename: str
    status: str = "processing"


class UploadStatusResponse(BaseModel):
    document_id: str
    collection: str | None = None
    status: str  # processing | ready | failed | unknown
    embedding_status: str | None = None
    milvus_insert_status: str | None = None
    parent_chunks: int | None = None
    child_chunks: int | None = None
    error: str | None = None
    warnings: list[str] = []


class UploadedDoc(BaseModel):
    document_id: str
    document: str
    collection: str | None = None
    status: str  # processing | ready | failed | unknown
    child_chunks: int | None = None
    updated_at: str | None = None
    deletable: bool = False


class UploadedDocsResponse(BaseModel):
    documents: list[UploadedDoc]
    count: int


class DeleteResponse(BaseModel):
    document_id: str
    collection: str | None = None
    collection_dropped: bool = False
    status: str = "deleted"


class TraceResponse(BaseModel):
    trace: dict[str, Any]


class IngestionTracesResponse(BaseModel):
    traces: list[dict[str, Any]]
    count: int
    collection: str


class MetricsResponse(BaseModel):
    total_queries: int
    success_queries: int
    failed_queries: int
    error_rate: float
    success_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    fastest_latency_ms: float
    slowest_latency_ms: float
    avg_retrieval_ms: float
    avg_generation_ms: float
    avg_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    avg_tokens_per_sec: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_prompt_tokens: float = 0.0
    fallback_counts: dict[str, int]
    error_stage_counts: dict[str, int]
    error_code_counts: dict[str, int] = {}
    trace_dir: str
