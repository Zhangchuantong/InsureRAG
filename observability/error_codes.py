# -*- coding: utf-8 -*-
"""Central error-code mapping for InsureRAG."""

from __future__ import annotations

from typing import Any


PDF_PARSE_ERROR = "PDF_PARSE_ERROR"
OCR_ERROR = "OCR_ERROR"
MILVUS_SEARCH_ERROR = "MILVUS_SEARCH_ERROR"
RERANKER_ERROR = "RERANKER_ERROR"
LLM_TIMEOUT = "LLM_TIMEOUT"
CACHE_ERROR = "CACHE_ERROR"
EMBEDDING_ERROR = "EMBEDDING_ERROR"
LLM_GENERATION_ERROR = "LLM_GENERATION_ERROR"
UNKNOWN_ERROR = "UNKNOWN_ERROR"


STAGE_ERROR_CODES = {
    "pdf_parse": PDF_PARSE_ERROR,
    "ocr": OCR_ERROR,
    "dense_search": MILVUS_SEARCH_ERROR,
    "sparse_search": MILVUS_SEARCH_ERROR,
    "rrf": MILVUS_SEARCH_ERROR,
    "retrieval": MILVUS_SEARCH_ERROR,
    "milvus": MILVUS_SEARCH_ERROR,
    "rerank": RERANKER_ERROR,
    "generation": LLM_GENERATION_ERROR,
    "llm": LLM_GENERATION_ERROR,
    "cache": CACHE_ERROR,
    "embedding": EMBEDDING_ERROR,
}


def is_timeout_error(error: BaseException | str) -> bool:
    name = error.__class__.__name__ if isinstance(error, BaseException) else ""
    message = str(error).lower()
    return "timeout" in name.lower() or "timed out" in message or "timeout" in message


def classify_error(stage: str, error: BaseException | str | None = None) -> str:
    if error is not None and is_timeout_error(error):
        return LLM_TIMEOUT if stage in {"generation", "llm"} else UNKNOWN_ERROR
    return STAGE_ERROR_CODES.get(stage, UNKNOWN_ERROR)


def first_error_code(errors: list[dict[str, Any]] | None) -> str | None:
    if not errors:
        return None
    first = errors[0]
    return first.get("code") or classify_error(first.get("stage", "unknown"), first.get("message", ""))
