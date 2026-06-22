# -*- coding: utf-8 -*-
"""MCP tools for InsureRAG."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from config.settings import settings
from services.rag_service import (
    get_clause_detail as service_get_clause_detail,
    list_documents as service_list_documents,
    query_insurance_clause as service_query_insurance_clause,
    search_related_clauses as service_search_related_clauses,
)

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "InsureRAG",
    instructions=(
        "Insurance clause retrieval and question answering tools. "
        "All outputs are structured JSON dictionaries."
    ),
)


def _error_response(error: Exception) -> dict[str, Any]:
    logger.exception("MCP tool failed: %s", error)
    return {
        "ok": False,
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
        },
    }


@mcp.tool()
def query_insurance_clause(question: str, collection: str = "") -> dict[str, Any]:
    """Answer an insurance clause question and return evidence plus trace_id."""
    try:
        result = service_query_insurance_clause(
            question=question,
            collection=collection or settings.vector_db["collection"],
        )
        result["ok"] = True
        return result
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
def search_related_clauses(query: str, top_k: int = 5, collection: str = "") -> dict[str, Any]:
    """Search related insurance clauses without generating an answer."""
    try:
        result = service_search_related_clauses(
            query=query,
            top_k=top_k,
            collection=collection or settings.vector_db["collection"],
        )
        result["ok"] = True
        return result
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
def get_clause_detail(clause_id: str, collection: str = "") -> dict[str, Any]:
    """Get a full parent clause by clause_id, parent_id, or article label."""
    try:
        result = service_get_clause_detail(
            clause_id,
            collection=collection or settings.vector_db["collection"],
        )
        result["ok"] = True
        return result
    except Exception as exc:
        return _error_response(exc)


@mcp.tool()
def list_documents() -> dict[str, Any]:
    """List documents currently loaded into the local knowledge base."""
    try:
        result = service_list_documents()
        result["ok"] = True
        return result
    except Exception as exc:
        return _error_response(exc)
