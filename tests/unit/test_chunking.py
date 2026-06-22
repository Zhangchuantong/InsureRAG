# -*- coding: utf-8 -*-

from chunking.parent_child_chunker import build_chunks, split_by_article


def test_split_by_article_keeps_clause_id():
    parents = split_by_article("第一条 合同构成。内容A。第二条 投保范围。内容B。")

    assert parents[0]["clause_id"] == "第一条"
    assert parents[1]["clause_id"] == "第二条"


def test_child_chunk_can_trace_back_to_parent():
    chunks = build_chunks(
        [{"source": "demo.pdf", "text": "第三条 等待期。等待期为90天。", "document_id": "demo"}],
        collection_name="insure_rag",
        chunk_size=20,
        chunk_overlap=0,
    )

    assert chunks
    assert chunks[0]["clause_id"] == "第三条"
    assert chunks[0]["parent_id"].startswith("demo:p")
    assert "等待期" in chunks[0]["parent_text"]


def test_short_parent_keeps_single_parent_id():
    chunks = build_chunks(
        [{"source": "s.pdf", "text": "第一条 等待期为90天。", "document_id": "s"}],
        collection_name="insure_rag",
        max_parent_chars=1000,
    )
    # short clause is not sub-split -> original parent_id scheme, no _suffix
    assert {c["parent_id"] for c in chunks} == {"s:p0"}


def test_oversized_parent_is_split_into_subparents():
    glossary = "第一条 释义。" + "".join(
        f"【术语{i}】这是第{i}个术语的较为详细的定义说明文字内容。" for i in range(60)
    )
    chunks = build_chunks(
        [{"source": "g.pdf", "text": glossary, "document_id": "g"}],
        collection_name="insure_rag",
        max_parent_chars=300,
    )
    parent_ids = {c["parent_id"] for c in chunks}
    # one clause becomes several bounded sub-parents
    assert len(parent_ids) > 1
    assert any("_" in pid for pid in parent_ids)
    # all sub-parents keep the same clause_id (citations stay correct)
    assert {c["clause_id"] for c in chunks} == {"第一条"}
    # each sub-parent text is bounded (not the whole glossary)
    assert all(len(c["parent_text"]) <= 450 for c in chunks)
