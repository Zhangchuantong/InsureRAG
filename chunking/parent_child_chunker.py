# -*- coding: utf-8 -*-
"""Parent-child chunking for insurance clauses."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

ARTICLE_PATTERN = re.compile(r"(?=第[一二三四五六七八九十百零两\d]+条)")
ARTICLE_TITLE_PATTERN = re.compile(r"第[一二三四五六七八九十百零两\d]+条")


def clean_repeated_pdf_text(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        previous = None
        while previous != line:
            previous = line
            line = re.sub(r"(.{2,40}?)\1{2,3}", r"\1", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def split_by_article(text: str) -> list[dict[str, str]]:
    """Split text by Chinese article headings and keep clause_id metadata."""
    text = clean_repeated_pdf_text(text)
    parts = [part.strip() for part in ARTICLE_PATTERN.split(text) if part.strip()]
    if not parts and text.strip():
        parts = [text.strip()]

    parents = []
    for index, parent_text in enumerate(parts):
        match = ARTICLE_TITLE_PATTERN.search(parent_text)
        parents.append(
            {
                "clause_id": match.group(0) if match else f"parent_{index}",
                "parent_text": parent_text,
            }
        )
    return parents


def default_document_id(source: str) -> str:
    stem = Path(source).stem
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", stem).strip("_") or "document"


# Parents longer than this (in characters) are split into multiple sub-parents.
# Most clauses are well under this; only oversized ones (e.g. the 释义/definitions
# clause, which lists dozens of terms) get split, so a query that matches one
# term returns just the relevant section instead of the whole glossary.
DEFAULT_MAX_PARENT_CHARS = 1200


def _split_long_parent(parent_text: str, max_chars: int, overlap: int = 80) -> list[str]:
    """Split an over-long parent into bounded sub-parents at natural boundaries.

    Short parents are returned unchanged (single element), so normal clauses keep
    their original parent_id / parent_text exactly.
    """
    if max_chars <= 0 or len(parent_text) <= max_chars:
        return [parent_text]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        # prefer the 【term】 boundary used in 释义 clauses, then paragraphs/sentences
        separators=["\n【", "\n\n", "\n", "。", "；", "，", " ", ""],
    )
    pieces = [piece.strip() for piece in splitter.split_text(parent_text) if piece.strip()]
    return pieces or [parent_text]


def build_chunks(
    docs: list[tuple[str, str]] | list[dict[str, Any]],
    collection_name: str,
    document_type: str = "insurance_clause",
    chunk_size: int = 200,
    chunk_overlap: int = 30,
    max_parent_chars: int = DEFAULT_MAX_PARENT_CHARS,
) -> list[dict[str, Any]]:
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    chunk_id = 0
    for doc in docs:
        if isinstance(doc, dict):
            source = doc["source"]
            text = doc["text"]
            document_id = doc.get("document_id") or default_document_id(source)
            page = doc.get("page")
            doc_type = doc.get("document_type", document_type)
        else:
            source, text = doc
            document_id = default_document_id(source)
            page = None
            doc_type = document_type

        parents = split_by_article(text)
        for parent_index, parent in enumerate(parents):
            sub_parents = _split_long_parent(parent["parent_text"], max_parent_chars)
            multi = len(sub_parents) > 1
            for sub_index, sub_text in enumerate(sub_parents):
                parent_id = f"{document_id}:p{parent_index}"
                if multi:
                    parent_id = f"{parent_id}_{sub_index}"
                children = child_splitter.split_text(sub_text)
                for child_text in children:
                    chunks.append(
                        {
                            "id": chunk_id,
                            "collection_name": collection_name,
                            "document_id": document_id,
                            "document_type": doc_type,
                            "source": source,
                            "page": page,
                            "clause_id": parent["clause_id"],
                            "parent_id": parent_id,
                            "parent_text": sub_text,
                            "child_text": child_text,
                        }
                    )
                    chunk_id += 1
    return chunks
