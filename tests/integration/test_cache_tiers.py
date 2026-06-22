# -*- coding: utf-8 -*-
"""Tests for the three-tier cache routing in services/rag_service.py.

query_insurance_clause chooses one of:
  1. exact_answer                      - normalized-identical question
  2. semantic_answer_direct            - similarity >= direct threshold
  3. semantic_answer_evidence_verified - similarity >= answer threshold AND
                                         re-checked evidence overlaps enough
  4. semantic_retrieval                - similarity >= retrieval threshold,
                                         reuse cached evidence, regenerate
  5. miss                              - full RAG

All external dependencies (cache, retrieval, generation, trace store) are
stubbed so each branch can be driven deterministically.
"""

import pytest

import services.rag_service as rs


@pytest.fixture
def tier_env(monkeypatch):
    cache = rs.settings.cache
    monkeypatch.setitem(cache, "answer_cache", True)
    monkeypatch.setitem(cache, "semantic_cache", True)
    monkeypatch.setitem(cache, "semantic_direct_threshold", 0.98)
    monkeypatch.setitem(cache, "semantic_answer_threshold", 0.95)
    monkeypatch.setitem(cache, "semantic_retrieval_threshold", 0.93)
    monkeypatch.setitem(cache, "evidence_overlap_threshold", 0.6)
    monkeypatch.setitem(cache, "require_top1_evidence_match", True)

    # Safe defaults: nothing cached, writes are no-ops, retrieval/generation faked.
    monkeypatch.setattr(rs, "get_json", lambda key: None)
    monkeypatch.setattr(rs, "search_semantic_candidates", lambda *a, **k: [])
    monkeypatch.setattr(rs, "set_json", lambda *a, **k: None)
    monkeypatch.setattr(rs, "store_semantic_answer", lambda *a, **k: None)
    monkeypatch.setattr(rs, "save_trace", lambda *a, **k: None)
    monkeypatch.setattr(rs, "search", lambda *a, **k: [])
    monkeypatch.setattr(rs, "answer_with_trace", lambda *a, **k: {"answer": "FULL", "trace_id": "t-full"})
    monkeypatch.setattr(rs, "answer_with_hits_trace", lambda *a, **k: {"answer": "RETRIEVAL", "trace_id": "t-retr"})
    monkeypatch.setattr(
        rs,
        "_load_trace_safe",
        lambda trace_id: {
            "latency": {"total_ms": 1.0},
            "final_context": [],
            "metadata": {"generation_mode": "llm"},
            "errors": [],
        },
    )
    return monkeypatch


def _candidate(similarity: float) -> dict:
    return {
        "similarity": similarity,
        "answer_json": {"answer": "CACHED", "evidence": [], "trace_id": "c1"},
        "evidence": [{"collection_name": "insure_rag", "document_id": "d", "parent_id": "p1", "clause_id": "x"}],
    }


def test_tier1_exact_answer_short_circuits_semantic(tier_env):
    tier_env.setattr(rs, "get_json", lambda key: {"answer": "EXACT", "evidence": [], "trace_id": "c0"})

    semantic_called = {"hit": False}

    def _spy(*args, **kwargs):
        semantic_called["hit"] = True
        return []

    tier_env.setattr(rs, "search_semantic_candidates", _spy)

    out = rs.query_insurance_clause("等待期内能赔吗？")

    assert out["cache_hit"] is True
    assert out["cache_type"] == "exact_answer"
    assert out["answer"] == "EXACT"
    # exact hit must return before semantic lookup runs
    assert semantic_called["hit"] is False
    # cache hit must mint a NEW trace_id for this request, not reuse the cached one
    assert out["trace_id"] and out["trace_id"] != "c0"


def test_tier2_semantic_answer_direct(tier_env):
    tier_env.setattr(rs, "search_semantic_candidates", lambda *a, **k: [_candidate(0.99)])

    out = rs.query_insurance_clause("等待期内能赔吗？")

    assert out["cache_type"] == "semantic_answer_direct"
    assert out["answer"] == "CACHED"
    assert out["similarity"] == pytest.approx(0.99)
    assert out["trace_id"] and out["trace_id"] != "c1"


def test_tier2_evidence_verified_when_overlap_and_top1_pass(tier_env):
    tier_env.setattr(rs, "search_semantic_candidates", lambda *a, **k: [_candidate(0.96)])
    tier_env.setattr(rs, "evidence_overlap", lambda current, cached: (0.8, True))

    out = rs.query_insurance_clause("等待期内能赔吗？")

    assert out["cache_type"] == "semantic_answer_evidence_verified"
    assert out["answer"] == "CACHED"
    assert out["evidence_overlap"] == pytest.approx(0.8)
    assert out["top1_evidence_match"] is True


def test_tier2_evidence_verify_fails_falls_back_to_retrieval(tier_env):
    # similarity high enough for the answer-threshold branch, but evidence diverges
    tier_env.setattr(rs, "search_semantic_candidates", lambda *a, **k: [_candidate(0.96)])
    tier_env.setattr(rs, "evidence_overlap", lambda current, cached: (0.2, False))

    out = rs.query_insurance_clause("等待期内能赔吗？")

    assert out["cache_type"] == "semantic_retrieval"
    assert out["answer"] == "RETRIEVAL"


def test_tier3_semantic_retrieval_between_thresholds(tier_env):
    # 0.93 <= sim < 0.95 -> skip answer branch, reuse evidence + regenerate
    tier_env.setattr(rs, "search_semantic_candidates", lambda *a, **k: [_candidate(0.94)])

    out = rs.query_insurance_clause("等待期内能赔吗？")

    assert out["cache_type"] == "semantic_retrieval"
    assert out["answer"] == "RETRIEVAL"
    assert out["cache_hit"] is True


def test_tier4_miss_runs_full_rag(tier_env):
    out = rs.query_insurance_clause("一个全新的问题")

    assert out["cache_hit"] is False
    assert out["cache_type"] == "miss"
    assert out["answer"] == "FULL"
