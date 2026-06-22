# -*- coding: utf-8 -*-
"""Stage 2: embed chunks and insert them into Milvus Standalone."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pymilvus import DataType, MilvusClient

from config.settings import settings
from observability.ingestion_trace import update_ingestion_trace
from vectorstore.milvus_client import forget_collection

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
VECTOR_DB_URI = settings.milvus_uri
COLLECTION = settings.vector_db["collection"]


def configure_hf_offline() -> None:
    if bool(settings.embedding.get("local_files_only", True)):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


def get_embedding_model():
    print(f"【阶段2】加载 embedding 模型：{settings.embedding['model_path']}")
    configure_hf_offline()
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction

    return BGEM3EmbeddingFunction(
        model_name=settings.embedding["model_path"],
        use_fp16=bool(settings.embedding["use_fp16"]),
        device=settings.embedding["device"],
        local_files_only=bool(settings.embedding.get("local_files_only", True)),
    )


# Per-field VARCHAR byte limits. parent_text is raised to the Milvus ceiling
# (65535) because OCR'd documents without clean "第X条" boundaries can produce a
# single very large parent block. Milvus measures VARCHAR length in bytes, so the
# write path truncates by UTF-8 byte length to stay within these limits.
VARCHAR_LIMITS = {
    "collection_name": 128,
    "document_id": 256,
    "document_type": 128,
    "source": 512,
    "clause_id": 256,
    "parent_id": 256,
    "child_text": 8000,
    "parent_text": 65535,
}


def _truncate_utf8(text: object, max_bytes: int) -> str:
    encoded = str(text).encode("utf-8")
    if len(encoded) <= max_bytes:
        return str(text)
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def build_collection(client: MilvusClient, dense_dim: int, collection_name: str = COLLECTION) -> None:
    # NOTE: rebuild-style ingestion. Every run drops and recreates the whole
    # collection, so re-running stage2 re-indexes all documents from scratch and
    # does NOT support incremental per-document insert/update/delete. This keeps
    # the demo simple; production multi-document ingestion would need upsert by
    # document_id and partial re-indexing instead.
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("collection_name", DataType.VARCHAR, max_length=VARCHAR_LIMITS["collection_name"])
    schema.add_field("document_id", DataType.VARCHAR, max_length=VARCHAR_LIMITS["document_id"])
    schema.add_field("document_type", DataType.VARCHAR, max_length=VARCHAR_LIMITS["document_type"])
    schema.add_field("source", DataType.VARCHAR, max_length=VARCHAR_LIMITS["source"])
    schema.add_field("page", DataType.INT64)
    schema.add_field("clause_id", DataType.VARCHAR, max_length=VARCHAR_LIMITS["clause_id"])
    schema.add_field("parent_id", DataType.VARCHAR, max_length=VARCHAR_LIMITS["parent_id"])
    schema.add_field("child_text", DataType.VARCHAR, max_length=VARCHAR_LIMITS["child_text"])
    schema.add_field("parent_text", DataType.VARCHAR, max_length=VARCHAR_LIMITS["parent_text"])
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dense_dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    # the collection was just dropped+recreated; invalidate the search pool's
    # load cache so the next query loads the fresh collection.
    forget_collection(collection_name)
    print(f"  collection 创建完成：{collection_name}，dense_dim={dense_dim}")


def _document_ids(chunks: list[dict]) -> list[str]:
    return sorted({str(chunk.get("document_id", chunk.get("source", "document"))) for chunk in chunks})


def _mark_documents(document_ids: list[str], **updates) -> None:
    for document_id in document_ids:
        update_ingestion_trace(document_id, **updates)


def _row_from_chunk(chunk: dict, dense_vector, sparse_vector) -> dict:
    def field(name: str, value: object) -> str:
        return _truncate_utf8(value, VARCHAR_LIMITS[name])

    return {
        "id": int(chunk["id"]),
        "collection_name": field("collection_name", chunk.get("collection_name", COLLECTION)),
        "document_id": field("document_id", chunk.get("document_id", chunk.get("source", "document"))),
        "document_type": field("document_type", chunk.get("document_type", "")),
        "source": field("source", chunk.get("source", "")),
        "page": int(chunk.get("page") if chunk.get("page") is not None else -1),
        "clause_id": field("clause_id", chunk.get("clause_id", "")),
        "parent_id": field("parent_id", chunk.get("parent_id", "")),
        "child_text": field("child_text", chunk["child_text"]),
        "parent_text": field("parent_text", chunk["parent_text"]),
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector,
    }


def index_chunks(chunks: list[dict], collection_name: str = COLLECTION) -> dict:
    """Embed chunks and (re)build the target collection. Shared by the CLI and
    the upload ingestion path. The collection is dropped and recreated, which is
    exactly what we want for a per-document upload collection (re-upload replaces
    only that document's own collection, never touching others)."""
    if not chunks:
        raise ValueError("No chunks to index.")

    document_ids = _document_ids(chunks)
    bge = get_embedding_model()
    texts = [chunk["child_text"] for chunk in chunks]
    print(f"【阶段2】生成 dense + sparse embeddings... collection={collection_name}")
    try:
        embeddings = bge(texts)
        _mark_documents(document_ids, embedding_status="success")
    except Exception as exc:
        _mark_documents(document_ids, embedding_status="failed", error=f"embedding failed: {exc}")
        raise

    dense_vecs = embeddings["dense"]
    sparse_vecs = embeddings["sparse"]
    dense_dim = len(dense_vecs[0])

    client = MilvusClient(uri=VECTOR_DB_URI)
    build_collection(client, dense_dim, collection_name=collection_name)

    rows = [
        _row_from_chunk(chunk, dense_vecs[index], sparse_vecs[[index]])
        for index, chunk in enumerate(chunks)
    ]

    try:
        client.insert(collection_name=collection_name, data=rows)
        client.flush(collection_name)
        _mark_documents(document_ids, milvus_insert_status="success")
    except Exception as exc:
        _mark_documents(document_ids, milvus_insert_status="failed", error=f"milvus insert failed: {exc}")
        raise

    print(f"  已插入 {len(rows)} 条数据到 Milvus：{VECTOR_DB_URI} / {collection_name}")
    return {"collection": collection_name, "chunk_count": len(rows), "document_ids": document_ids}


def main() -> None:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"【阶段2】读取 {len(chunks)} 个子块")
    index_chunks(chunks, collection_name=COLLECTION)
    print("  阶段2完成，可运行 stage3_search.py 测试检索。")


if __name__ == "__main__":
    main()
