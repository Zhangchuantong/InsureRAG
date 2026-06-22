# -*- coding: utf-8 -*-
"""Unit tests for the cache primitives in cache/cache_store.py.

These exercise the pure logic (normalization, similarity, evidence overlap,
key construction) plus the JSON backend round-trip / TTL / semantic guards,
without requiring Redis or loading the embedding model.
"""

import shutil
import uuid
from pathlib import Path

import pytest

import cache.cache_store as cs


@pytest.fixture
def json_cache(monkeypatch):
    """Force the JSON backend at an isolated, project-local temp path.

    Deliberately avoids pytest's ``tmp_path`` / ``--basetemp`` machinery: on some
    Windows setups those pytest-managed temp dirs end up Access-denied. We create
    a plain project-local directory instead and clean it up afterwards.
    """
    base = Path(__file__).resolve().parents[2] / ".test_artifacts" / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    cache_file = base / "cache.json"
    monkeypatch.setitem(cs.settings.cache, "enabled", True)
    monkeypatch.setitem(cs.settings.cache, "backend", "json")
    monkeypatch.setitem(cs.settings.cache, "path", str(cache_file))
    monkeypatch.setitem(cs.settings.cache, "semantic_cache", True)
    monkeypatch.setitem(cs.settings.cache, "ttl_seconds", 3600)
    yield cache_file
    shutil.rmtree(base, ignore_errors=True)


def _evidence(parent_id: str, clause: str = "x", collection: str = "insure_rag") -> dict:
    return {
        "collection_name": collection,
        "document_id": "d",
        "parent_id": parent_id,
        "clause_id": clause,
    }


def _semantic_result(top_k: int = 5, collection: str = "insure_rag") -> dict:
    return {
        "answer": "cached answer",
        "evidence": [_evidence("p1")],
        "trace_id": "t1",
        "latency_ms": 1.0,
        "collection": collection,
        "top_k": top_k,
    }


# --- pure helpers ----------------------------------------------------------


def test_normalize_query_unifies_punctuation_and_whitespace():
    assert cs.normalize_query("  等待期内生病了能赔吗？ ") == "等待期内生病了能赔吗?"
    assert cs.normalize_query("A，B。C；") == "a,b.c;"
    assert cs.normalize_query("a   b\tc\n") == "a b c"
    assert cs.normalize_query("") == ""


def test_cosine_similarity_edge_cases():
    assert cs.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cs.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cs.cosine_similarity([], [1.0]) == 0.0           # empty
    assert cs.cosine_similarity([1.0, 2.0], [1.0]) == 0.0   # length mismatch
    assert cs.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero norm


def test_evidence_overlap_jaccard_and_top1_match():
    current = [_evidence("p1"), _evidence("p2")]
    cached = [_evidence("p1"), _evidence("p3")]
    overlap, top1 = cs.evidence_overlap(current, cached)
    # intersection {p1} over union {p1,p2,p3} = 1/3, and top1 ids match
    assert overlap == pytest.approx(1 / 3)
    assert top1 is True


def test_evidence_overlap_top1_mismatch_and_empty():
    current = [_evidence("p2"), _evidence("p1")]
    cached = [_evidence("p1"), _evidence("p2")]
    overlap, top1 = cs.evidence_overlap(current, cached)
    assert overlap == pytest.approx(1.0)   # same set
    assert top1 is False                   # different first element
    assert cs.evidence_overlap([], cached) == (0.0, False)


def test_answer_cache_key_normalizes_and_is_sensitive_to_inputs():
    base = cs.answer_cache_key("等待期？", "insure_rag", 5)
    # normalization makes whitespace / fullwidth punctuation irrelevant
    assert base == cs.answer_cache_key("  等待期?  ", "insure_rag", 5)
    # but collection / top_k changes must change the key
    assert base != cs.answer_cache_key("等待期？", "insure_rag", 3)
    assert base != cs.answer_cache_key("等待期？", "other_collection", 5)
    assert base.startswith("answer:insure_rag:")


# --- JSON backend kv round-trip & TTL --------------------------------------


def test_get_set_json_roundtrip(json_cache):
    cs.set_json("k1", {"answer": "hi"}, cache_type="answer")
    assert cs.get_json("k1") == {"answer": "hi"}
    assert cs.get_json("missing") is None


def test_get_json_respects_ttl_expiry(json_cache, monkeypatch):
    cs.set_json("k1", {"a": 1}, cache_type="answer", ttl_seconds=100)
    now = cs.time.time()
    # jump past the TTL window
    monkeypatch.setattr(cs.time, "time", lambda: now + 200)
    assert cs.get_json("k1") is None


def test_cache_disabled_is_a_noop(json_cache, monkeypatch):
    monkeypatch.setitem(cs.settings.cache, "enabled", False)
    cs.set_json("k1", {"a": 1}, cache_type="answer")
    assert cs.get_json("k1") is None


# --- semantic store / search ----------------------------------------------


def test_semantic_store_and_search_by_similarity(json_cache, monkeypatch):
    vectors = {"q_store": [1.0, 0.0], "q_same": [1.0, 0.0], "q_orthogonal": [0.0, 1.0]}
    monkeypatch.setattr(cs, "_query_embedding", lambda q: vectors[q])

    cs.store_semantic_answer("q_store", _semantic_result(), generation_mode="llm", has_errors=False)

    hits = cs.search_semantic_candidates("q_same", collection="insure_rag", top_k=5, min_similarity=0.93)
    assert len(hits) == 1
    assert hits[0]["similarity"] == pytest.approx(1.0)
    assert hits[0]["answer_json"]["answer"] == "cached answer"

    # orthogonal vector is below threshold -> no candidate
    assert cs.search_semantic_candidates("q_orthogonal", collection="insure_rag", top_k=5, min_similarity=0.93) == []


def test_semantic_search_guards_on_collection_and_top_k(json_cache, monkeypatch):
    monkeypatch.setattr(cs, "_query_embedding", lambda q: [1.0, 0.0])
    cs.store_semantic_answer("q", _semantic_result(top_k=5, collection="insure_rag"),
                             generation_mode="llm", has_errors=False)

    # identical vector (cos=1.0) but mismatched collection / top_k must be rejected
    assert cs.search_semantic_candidates("q", collection="other", top_k=5, min_similarity=0.5) == []
    assert cs.search_semantic_candidates("q", collection="insure_rag", top_k=3, min_similarity=0.5) == []
    # the matching collection + top_k still returns it
    assert len(cs.search_semantic_candidates("q", collection="insure_rag", top_k=5, min_similarity=0.5)) == 1


def test_store_semantic_answer_skips_non_cacheable_results(json_cache, monkeypatch):
    monkeypatch.setattr(cs, "_query_embedding", lambda q: [1.0, 0.0])
    base = _semantic_result()

    cs.store_semantic_answer("q", base, generation_mode="evidence_fallback", has_errors=False)
    cs.store_semantic_answer("q", base, generation_mode="llm", has_errors=True)
    cs.store_semantic_answer("q", {**base, "evidence": []}, generation_mode="llm", has_errors=False)

    # none of the above should have been persisted
    assert cs.search_semantic_candidates("q", collection="insure_rag", top_k=5, min_similarity=0.0) == []
