# -*- coding: utf-8 -*-
"""Unit tests for VARCHAR byte-safe truncation in the indexing path."""

import stage2_build_db as s2


def test_truncate_keeps_short_text_unchanged():
    assert s2._truncate_utf8("第三条 等待期为90天", 1000) == "第三条 等待期为90天"


def test_truncate_respects_byte_limit_for_chinese():
    text = "保" * 5000  # 5000 chars = 15000 UTF-8 bytes
    out = s2._truncate_utf8(text, 8000)
    encoded = out.encode("utf-8")
    assert len(encoded) <= 8000
    # must remain valid UTF-8 (no half multibyte char at the cut point)
    encoded.decode("utf-8")


def test_row_truncates_oversized_parent_text_to_schema_limit():
    chunk = {
        "id": 1,
        "child_text": "等待期",
        "parent_text": "保" * 30000,  # ~90000 bytes, exceeds 65535
        "source": "s.pdf",
        "clause_id": "第三条",
        "parent_id": "d:p0",
        "document_id": "d",
    }
    row = s2._row_from_chunk(chunk, [0.1, 0.2], None)
    assert len(row["parent_text"].encode("utf-8")) <= s2.VARCHAR_LIMITS["parent_text"]
    assert len(row["child_text"].encode("utf-8")) <= s2.VARCHAR_LIMITS["child_text"]
