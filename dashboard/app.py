# -*- coding: utf-8 -*-
"""Minimal dashboard for overview, document ingestion traces, and query traces."""

from __future__ import annotations

import html
import json

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse

from api.auth import require_api_key
from config.settings import settings
from observability import trace_db
from observability.metrics_store import calculate_metrics
from observability.trace_store import load_trace
from services.rag_service import list_documents, list_ingestion_traces


# Dashboard exposes traces (questions/answers/evidence) and metrics. Gate it with
# the same optional API key as the main API: open by default, required when
# api.auth_enabled is true. For a local-only dashboard, also prefer binding to
# 127.0.0.1 instead of exposing 0.0.0.0 publicly.
app = FastAPI(title="InsureRAG Dashboard", dependencies=[Depends(require_api_key)])


def _layout(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    code, pre {{ background: #f7f7f7; padding: 2px 4px; }}
    .nav a {{ margin-right: 16px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="nav">
    <a href="/">Overview</a>
    <a href="/documents">Document Browser</a>
    <a href="/traces">Query Traces</a>
  </div>
  {body}
</body>
</html>"""
    )


@app.get("/", response_class=HTMLResponse)
def overview() -> HTMLResponse:
    docs = list_documents()
    ingestion = list_ingestion_traces()
    trace_count = trace_db.count_query_traces()
    metrics = calculate_metrics()
    cache = docs.get("cache", {})
    fallback_rows = "".join(
        f"<li>{html.escape(str(name))}: {count}</li>"
        for name, count in metrics.get("fallback_counts", {}).items()
    ) or "<li>None</li>"
    error_rows = "".join(
        f"<li>{html.escape(str(name))}: {count}</li>"
        for name, count in metrics.get("error_stage_counts", {}).items()
    ) or "<li>None</li>"
    body = f"""
<h2>Overview</h2>
<ul>
  <li>Collection: <code>{html.escape(docs['collection'])}</code></li>
  <li>Documents: {len(docs['documents'])}</li>
  <li>Chunks: {docs['chunk_count']}</li>
  <li>Ingestion Traces: {ingestion['count']}</li>
  <li>Query Traces: {trace_count}</li>
  <li>LLM: <code>{html.escape(settings.llm['model'])}</code></li>
  <li>Milvus: <code>{html.escape(settings.milvus_uri)}</code></li>
  <li>Cache: <code>{html.escape(str(cache.get('backend', 'disabled')))}</code></li>
  <li>Exact Cache Entries: {cache.get('kv_cache_count', 0)}</li>
  <li>Semantic Cache Entries: {cache.get('semantic_cache_count', 0)}</li>
</ul>
<h2>Metrics</h2>
<table>
  <tr><th>Total Queries</th><td>{metrics['total_queries']}</td></tr>
  <tr><th>Success Queries</th><td>{metrics['success_queries']}</td></tr>
  <tr><th>Failed Queries</th><td>{metrics['failed_queries']}</td></tr>
  <tr><th>Success Rate</th><td>{metrics['success_rate']:.2%}</td></tr>
  <tr><th>Error Rate</th><td>{metrics['error_rate']:.2%}</td></tr>
  <tr><th>Avg Latency</th><td>{metrics['avg_latency_ms']} ms</td></tr>
  <tr><th>P50 Latency</th><td>{metrics['p50_latency_ms']} ms</td></tr>
  <tr><th>P95 Latency</th><td>{metrics['p95_latency_ms']} ms</td></tr>
  <tr><th>Avg Retrieval</th><td>{metrics['avg_retrieval_ms']} ms</td></tr>
  <tr><th>Avg Generation</th><td>{metrics['avg_generation_ms']} ms</td></tr>
</table>
<h3>Fallback Counts</h3>
<ul>{fallback_rows}</ul>
<h3>Error Stage Counts</h3>
<ul>{error_rows}</ul>
"""
    return _layout("InsureRAG Dashboard", body)


@app.get("/documents", response_class=HTMLResponse)
def document_browser() -> HTMLResponse:
    traces = list_ingestion_traces()["traces"]
    rows = []
    for item in traces:
        warnings = "<br>".join(html.escape(str(w)) for w in item.get("warnings", []))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('document', '')))}</td>"
            f"<td>{html.escape(str(item.get('document_id', '')))}</td>"
            f"<td>{html.escape(str(item.get('parse_method', '')))}</td>"
            f"<td>{html.escape(str(item.get('ocr_used', False)))}</td>"
            f"<td>{float(item.get('quality_score') or 0):.3f}</td>"
            f"<td>{float(item.get('text_density') or 0):.1f}</td>"
            f"<td>{html.escape(str(item.get('page_count', 0)))}</td>"
            f"<td>{html.escape(str(item.get('parent_chunks', 0)))}</td>"
            f"<td>{html.escape(str(item.get('child_chunks', 0)))}</td>"
            f"<td>{html.escape(str(item.get('embedding_status', '')))}</td>"
            f"<td>{html.escape(str(item.get('milvus_insert_status', '')))}</td>"
            f"<td>{warnings}</td>"
            "</tr>"
        )
    body = """
<h2>Document Browser</h2>
<table>
  <tr>
    <th>Document</th><th>Document ID</th><th>Parse</th><th>OCR</th>
    <th>Quality</th><th>Density</th><th>Pages</th><th>Parents</th><th>Children</th>
    <th>Embedding</th><th>Milvus</th><th>Warnings</th>
  </tr>
""" + "\n".join(rows) + "\n</table>"
    return _layout("Document Browser", body)


@app.get("/traces", response_class=HTMLResponse)
def query_traces() -> HTMLResponse:
    rows = []
    for item in trace_db.list_query_traces(limit=50):
        trace_id = str(item.get("trace_id", ""))
        rows.append(
            "<tr>"
            f"<td><a href='/traces/{html.escape(trace_id)}'>{html.escape(trace_id)}</a></td>"
            f"<td>{html.escape(str(item.get('query', ''))[:120])}</td>"
            f"<td>{html.escape(str(item.get('total_ms', 0)))}</td>"
            f"<td>{html.escape(str(item.get('created_at', '')))}</td>"
            "</tr>"
        )
    body = """
<h2>Query Traces</h2>
<table><tr><th>Trace ID</th><th>Query</th><th>Total ms</th><th>Created</th></tr>
""" + "\n".join(rows) + "\n</table>"
    return _layout("Query Traces", body)


@app.get("/traces/{trace_id}", response_class=HTMLResponse)
def query_trace_detail(trace_id: str) -> HTMLResponse:
    try:
        data = load_trace(trace_id)
        pretty = html.escape(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as exc:
        pretty = html.escape(str(exc))
    return _layout("Query Trace Detail", f"<pre>{pretty}</pre>")


if __name__ == "__main__":
    uvicorn.run(
        "dashboard.app:app",
        host=settings.dashboard["host"],
        port=int(settings.dashboard["port"]),
        reload=False,
    )
