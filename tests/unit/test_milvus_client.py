# -*- coding: utf-8 -*-
"""Unit tests for the Milvus client reset + single-retry helpers."""

import pytest

import vectorstore.milvus_client as mc


def test_is_connection_error_classifies_transport_failures():
    assert mc.is_connection_error(ConnectionError("x"))
    assert mc.is_connection_error(TimeoutError("x"))
    assert mc.is_connection_error(RuntimeError("failed to connect to milvus server"))
    assert mc.is_connection_error(RuntimeError("rpc deadline exceeded"))
    # a logical/schema error is NOT a connection error
    assert not mc.is_connection_error(ValueError("length of varchar field exceeds max length"))


def test_reset_clears_client_and_loaded_collections():
    mc._client = object()
    mc._loaded_collections.add("some_collection")
    mc.reset_milvus_client()
    assert mc._client is None
    assert "some_collection" not in mc._loaded_collections


def test_run_with_milvus_retry_resets_and_retries_once(monkeypatch):
    state = {"calls": 0, "resets": 0}
    monkeypatch.setattr(mc, "reset_milvus_client", lambda: state.__setitem__("resets", state["resets"] + 1))

    def operation():
        state["calls"] += 1
        if state["calls"] == 1:
            raise ConnectionError("milvus unavailable")
        return "ok"

    assert mc.run_with_milvus_retry(operation) == "ok"
    assert state["calls"] == 2      # original + exactly one retry
    assert state["resets"] == 1     # reset happened between attempts


def test_run_with_milvus_retry_does_not_loop_forever(monkeypatch):
    state = {"calls": 0}
    monkeypatch.setattr(mc, "reset_milvus_client", lambda: None)

    def operation():
        state["calls"] += 1
        raise ConnectionError("still down")

    with pytest.raises(ConnectionError):
        mc.run_with_milvus_retry(operation)
    assert state["calls"] == 2      # tried twice, then gave up (no infinite retry)


def test_run_with_milvus_retry_propagates_non_connection_error(monkeypatch):
    state = {"calls": 0, "resets": 0}
    monkeypatch.setattr(mc, "reset_milvus_client", lambda: state.__setitem__("resets", state["resets"] + 1))

    def operation():
        state["calls"] += 1
        raise ValueError("varchar too long")

    with pytest.raises(ValueError):
        mc.run_with_milvus_retry(operation)
    assert state["calls"] == 1      # no retry for non-connection errors
    assert state["resets"] == 0
