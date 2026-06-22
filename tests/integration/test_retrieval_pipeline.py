# -*- coding: utf-8 -*-

from observability.trace_context import TraceContext
import stage3_search


class _Sparse:
    def __getitem__(self, _):
        return [0.1]


class _FakeBGE:
    def __call__(self, _texts):
        return {"dense": [[0.1, 0.2]], "sparse": _Sparse()}


class _FakeClient:
    def load_collection(self, _collection):
        return None

    def search(self, collection_name, data, anns_field, search_params, limit, output_fields):
        hit = {
            "id": 1 if anns_field == "dense_vector" else 2,
            "distance": 0.9,
            "entity": {
                "collection_name": collection_name,
                "document_id": "demo",
                "document_type": "insurance_clause",
                "source": "demo.pdf",
                "page": 1,
                "clause_id": "第三条",
                "parent_id": "demo:p0",
                "child_text": "等待期",
                "parent_text": "第三条 等待期为90天。",
            },
        }
        return [[hit]]


def test_retrieval_pipeline_with_mocked_dense_sparse(monkeypatch):
    monkeypatch.setattr(stage3_search, "get_bge", lambda: _FakeBGE())
    monkeypatch.setattr(stage3_search, "get_milvus_client", lambda: _FakeClient())
    monkeypatch.setattr(stage3_search, "load_collection_once", lambda _collection: None)
    monkeypatch.setitem(stage3_search.settings.reranker, "enabled", False)

    trace = TraceContext(query="等待期多久")
    results = stage3_search.search("等待期多久", top_n=2, trace=trace, collection_name="insure_rag")

    assert results
    assert trace.metadata["retrieval_mode"] == "hybrid_rrf"
    assert trace.metadata["rerank_mode"] == "disabled_rrf_topk"


def test_search_resets_and_retries_once_on_milvus_connection_error(monkeypatch):
    import vectorstore.milvus_client as mc

    counters = {"load": 0, "reset": 0}

    def flaky_load(_collection):
        counters["load"] += 1
        if counters["load"] == 1:
            raise ConnectionError("milvus unavailable")  # connection failure on first attempt

    monkeypatch.setattr(stage3_search, "get_bge", lambda: _FakeBGE())
    monkeypatch.setattr(stage3_search, "get_milvus_client", lambda: _FakeClient())
    monkeypatch.setattr(stage3_search, "load_collection_once", flaky_load)
    monkeypatch.setattr(mc, "reset_milvus_client", lambda: counters.__setitem__("reset", counters["reset"] + 1))
    monkeypatch.setitem(stage3_search.settings.reranker, "enabled", False)

    results = stage3_search.search("等待期", top_n=2, collection_name="insure_rag")

    assert results                  # retry succeeded -> results returned
    assert counters["load"] == 2    # retried exactly once
    assert counters["reset"] == 1   # client was reset between attempts


def test_search_degrades_gracefully_when_milvus_stays_down(monkeypatch):
    monkeypatch.setattr(stage3_search, "get_bge", lambda: _FakeBGE())
    monkeypatch.setattr(stage3_search, "get_milvus_client", lambda: _FakeClient())
    monkeypatch.setattr(stage3_search, "load_collection_once",
                        lambda _c: (_ for _ in ()).throw(ConnectionError("down")))
    monkeypatch.setitem(stage3_search.settings.reranker, "enabled", False)

    trace = TraceContext(query="等待期")
    # retry also fails -> no crash, empty results, error recorded on the trace
    results = stage3_search.search("等待期", top_n=2, trace=trace, collection_name="insure_rag")

    assert results == []
    assert trace.metadata.get("retrieval_mode") == "milvus_failed"


def test_reranker_failure_fallback_to_rrf(monkeypatch):
    class BadReranker:
        def predict(self, _pairs):
            raise RuntimeError("reranker down")

    monkeypatch.setitem(stage3_search.settings.reranker, "enabled", True)
    monkeypatch.setattr(stage3_search, "get_reranker", lambda: BadReranker())

    trace = TraceContext(query="等待期")
    trace.rrf_results = [{"score": 0.5}]
    candidate = {
        "id": 1,
        "distance": 0.1,
        "entity": {
            "child_text": "等待期",
            "parent_text": "第三条 等待期为90天。",
            "source": "demo.pdf",
        },
    }

    ranked = stage3_search.rerank("等待期", [candidate], top_n=1, trace=trace)

    assert ranked
    assert trace.metadata["rerank_mode"] == "rrf_topk_fallback"
