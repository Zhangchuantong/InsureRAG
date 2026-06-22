# -*- coding: utf-8 -*-

from stage3_search import _dedup_ranked_by_parent, _manual_rrf


def _hit(hit_id: int, text: str):
    return {
        "id": hit_id,
        "distance": 1.0,
        "entity": {
            "child_text": text,
            "parent_text": text,
            "source": "demo.pdf",
        },
    }


def _parent_hit(hit_id: int, parent_id: str, child_text: str, score: float):
    return (
        {
            "id": hit_id,
            "distance": score,
            "entity": {
                "collection_name": "insure_rag",
                "document_id": "demo",
                "source": "demo.pdf",
                "clause_id": "第三条",
                "parent_id": parent_id,
                "child_text": child_text,
                "parent_text": f"{parent_id} full parent clause",
            },
        },
        score,
    )


def test_rrf_fuses_dense_and_sparse_rankings():
    dense = [_hit(1, "A"), _hit(2, "B")]
    sparse = [_hit(2, "B"), _hit(3, "C")]

    fused = _manual_rrf(dense, sparse, rrf_k=60, limit=3)

    assert fused[0][0]["id"] == 2
    assert len(fused) == 3


def test_rrf_merges_duplicate_documents():
    dense = [_hit(1, "A")]
    sparse = [_hit(1, "A")]

    fused = _manual_rrf(dense, sparse, rrf_k=60, limit=5)

    assert len(fused) == 1
    assert fused[0][1] > 1 / 61


def test_parent_dedup_keeps_best_child_per_parent():
    ranked = [
        _parent_hit(1, "demo:p0", "等待期为90天", 0.95),
        _parent_hit(2, "demo:p0", "等待期内疾病不赔", 0.91),
        _parent_hit(3, "demo:p1", "酒后驾驶免责", 0.88),
    ]

    deduped, duplicate_count = _dedup_ranked_by_parent(ranked, top_n=5)

    assert duplicate_count == 1
    assert len(deduped) == 2
    assert deduped[0][0]["id"] == 1
    assert deduped[1][0]["entity"]["parent_id"] == "demo:p1"
