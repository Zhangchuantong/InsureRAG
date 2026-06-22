# -*- coding: utf-8 -*-
"""Unit tests for streaming generation metrics (TTFT / tokens / throughput).

The OpenAI client stream is faked so we can test the parsing + metric math
without a running vLLM server.
"""

import stage4_generate as g


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content=None, usage=None):
        self.choices = [_Choice(content)] if content is not None else []
        self.usage = usage


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def _fake_client(chunks):
    class _Completions:
        @staticmethod
        def create(**kwargs):
            assert kwargs.get("stream") is True
            return iter(chunks)

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


def test_call_vllm_reports_ttft_tokens_and_throughput(monkeypatch):
    chunks = [
        _Chunk("等待"),
        _Chunk("期为"),
        _Chunk("90天。"),
        _Chunk(content=None, usage=_Usage(prompt_tokens=120, completion_tokens=8)),
    ]
    monkeypatch.setattr(g, "client", _fake_client(chunks))

    res = g.call_vllm("等待期多久？")

    assert res.answer == "等待期为90天。"
    assert res.prompt_tokens == 120
    assert res.completion_tokens == 8
    assert res.ttft_ms is not None and res.ttft_ms >= 0
    assert res.generation_ms >= 0
    assert res.tokens_per_sec and res.tokens_per_sec > 0


def test_call_vllm_approximates_tokens_when_usage_missing(monkeypatch):
    # server that does not return usage -> fall back to streamed-chunk count
    chunks = [_Chunk("甲"), _Chunk("乙"), _Chunk("丙")]
    monkeypatch.setattr(g, "client", _fake_client(chunks))

    res = g.call_vllm("q")

    assert res.answer == "甲乙丙"
    assert res.prompt_tokens is None
    assert res.completion_tokens == 3  # approximated from 3 streamed chunks
    assert res.ttft_ms is not None
