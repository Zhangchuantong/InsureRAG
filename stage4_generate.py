# -*- coding: utf-8 -*-
"""Stage 4: generate answers with local vLLM/Qwen through OpenAI-compatible API."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from config.settings import settings
from observability.error_codes import LLM_GENERATION_ERROR, LLM_TIMEOUT, is_timeout_error
from observability.trace_context import TraceContext
from observability.trace_store import save_trace
from stage3_search import search

logger = logging.getLogger(__name__)

BASE_URL = settings.llm["base_url"]
API_KEY = settings.llm["api_key"]
MODEL = settings.llm["model"]
ENABLE_THINKING = bool(settings.llm["enable_thinking"])

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=float(settings.llm.get("timeout_seconds", 60)),
)

PROMPT_TEMPLATE = """你是一名严谨的保险条款问答助手。请只依据【召回条款】回答用户问题。

要求：
1. 如果召回条款能回答，直接给出结论，并尽量标注依据条款或来源。
2. 如果证据不足，请明确说明“根据当前召回条款，无法确定该问题”，不要编造。
3. 回答要简洁、清楚，避免输出无关内容。

【召回条款】
{context}

【用户问题】
{question}

【回答】
"""


def remove_thinking_text(text: str) -> str:
    """Remove Qwen thinking blocks from the returned answer."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    lower_cleaned = cleaned.lower()
    end_tag = "</think>"
    if end_tag in lower_cleaned:
        cleaned = cleaned[lower_cleaned.rfind(end_tag) + len(end_tag):]
    lower_cleaned = cleaned.lower()
    start_tag = "<think>"
    if start_tag in lower_cleaned:
        cleaned = cleaned[:lower_cleaned.find(start_tag)]
    return cleaned.strip()


@dataclass
class GenerationResult:
    """Answer plus per-call generation metrics captured from the token stream."""
    answer: str
    ttft_ms: float | None          # time to first token
    generation_ms: float           # full streaming duration
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_per_sec: float | None   # output throughput


def call_vllm(prompt: str) -> GenerationResult:
    """Stream the completion so we can measure first-token latency and throughput.

    Streaming (not the response text) is the only way to know *when* the first
    token arrives. token usage is requested via stream_options; if the server
    omits it we approximate completion tokens by the number of streamed chunks.
    """
    started_at = time.perf_counter()
    ttft: float | None = None
    pieces: list[str] = []
    usage = None

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一名严谨的保险条款问答助手，只能依据召回条款回答。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=float(settings.llm["temperature"]),
        max_tokens=int(settings.llm["max_tokens"]),
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": ENABLE_THINKING,
            }
        },
        stream=True,
        stream_options={"include_usage": True},
    )

    for event in stream:
        if getattr(event, "usage", None):
            usage = event.usage
        choices = getattr(event, "choices", None)
        if not choices:
            continue
        content = getattr(choices[0].delta, "content", None)
        if content:
            if ttft is None:
                ttft = time.perf_counter() - started_at
            pieces.append(content)

    generation_seconds = time.perf_counter() - started_at
    answer_text = remove_thinking_text("".join(pieces))

    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    if completion_tokens is None:
        completion_tokens = len(pieces) or None  # fallback: streamed-chunk count

    tokens_per_sec = None
    if completion_tokens and generation_seconds > 0:
        tokens_per_sec = round(completion_tokens / generation_seconds, 2)

    return GenerationResult(
        answer=answer_text,
        ttft_ms=round(ttft * 1000, 3) if ttft is not None else None,
        generation_ms=round(generation_seconds * 1000, 3),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_sec=tokens_per_sec,
    )


def _apply_generation_metrics(trace: TraceContext, gen: GenerationResult) -> None:
    trace.answer = gen.answer
    trace.metadata["generation_mode"] = "llm"
    if gen.ttft_ms is not None:
        trace.latency["ttft_ms"] = gen.ttft_ms
    trace.metadata["prompt_tokens"] = gen.prompt_tokens
    trace.metadata["completion_tokens"] = gen.completion_tokens
    trace.metadata["tokens_per_sec"] = gen.tokens_per_sec


def _build_prompt(question: str, hits: list[dict]) -> str:
    context_parts = []
    for hit in hits:
        source = hit.get("source", "未知来源")
        clause = hit.get("clause_id") or hit.get("clause") or ""
        parent_text = hit.get("parent_text", "")
        label = f"{source} {clause}".strip()
        context_parts.append(f"（来源：{label}）\n{parent_text}")
    context = "\n\n---\n\n".join(context_parts)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def _format_evidence_fallback(hits: list[dict], error: Exception) -> str:
    if not hits:
        return f"系统生成答案时发生异常，且当前没有可用检索证据。错误信息：{error}"

    snippets = []
    for index, hit in enumerate(hits[:3], start=1):
        parent_text = hit.get("parent_text", "").replace("\n", " ")
        source = hit.get("source", "未知来源")
        clause = hit.get("clause_id") or ""
        snippets.append(f"{index}. 来源：{source} {clause}；证据片段：{parent_text[:180]}...")
    return (
        "系统生成答案时发生异常，已降级返回检索到的证据片段，请人工核对后使用。\n"
        f"错误信息：{error}\n"
        "检索证据：\n"
        + "\n".join(snippets)
    )


def answer_with_trace(
    question: str,
    top_n: int | None = None,
    collection_name: str | None = None,
) -> dict[str, str]:
    collection = collection_name or settings.vector_db["collection"]
    trace = TraceContext(query=question)
    trace.metadata = {
        "top_k": max(int(settings.retrieval["dense_top_k"]), int(settings.retrieval["sparse_top_k"])),
        "final_k": top_n or int(settings.retrieval["final_top_k"]),
        "model": MODEL,
        "collection": collection,
    }

    try:
        hits = search(question, top_n=top_n, trace=trace, collection_name=collection)
    except Exception as exc:
        trace.add_error("retrieval", exc)
        hits = []

    started_at = time.perf_counter()
    try:
        prompt = _build_prompt(question, hits)
        if hits:
            _apply_generation_metrics(trace, call_vllm(prompt))
        else:
            trace.answer = "根据当前召回条款，无法确定该问题。建议补充相关保险条款后再查询。"
            trace.metadata["generation_mode"] = "no_context_fallback"
        trace.latency["generation_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
    except Exception as exc:
        trace.latency["generation_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
        logger.warning("LLM generation failed, fallback to retrieved evidence snippets: %s", exc)
        trace.add_error("generation", exc, code=LLM_TIMEOUT if is_timeout_error(exc) else LLM_GENERATION_ERROR)
        trace.add_warning("generation", "LLM generation failed; returned retrieved evidence snippets.")
        trace.metadata["generation_mode"] = "evidence_fallback"
        trace.answer = _format_evidence_fallback(hits, exc)

    try:
        save_trace(trace)
    except Exception as exc:
        logger.warning("Failed to save trace_id=%s: %s", trace.trace_id, exc)

    return {
        "answer": trace.answer,
        "trace_id": trace.trace_id,
    }


def answer_with_hits_trace(
    question: str,
    hits: list[dict],
    collection_name: str | None = None,
    metadata: dict | None = None,
) -> dict[str, str]:
    """Generate an answer from provided evidence without running retrieval again."""
    collection = collection_name or settings.vector_db["collection"]
    trace = TraceContext(query=question)
    trace.final_context = hits
    trace.metadata = {
        "top_k": len(hits),
        "final_k": len(hits),
        "model": MODEL,
        "collection": collection,
        "retrieval_mode": "cached_evidence",
        **(metadata or {}),
    }

    started_at = time.perf_counter()
    try:
        prompt = _build_prompt(question, hits)
        if hits:
            _apply_generation_metrics(trace, call_vllm(prompt))
        else:
            trace.answer = "根据当前召回条款，无法确定该问题。建议补充相关保险条款后再查询。"
            trace.metadata["generation_mode"] = "no_context_fallback"
        trace.latency["generation_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
    except Exception as exc:
        trace.latency["generation_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
        logger.warning("LLM generation failed with cached evidence: %s", exc)
        trace.add_error("generation", exc, code=LLM_TIMEOUT if is_timeout_error(exc) else LLM_GENERATION_ERROR)
        trace.add_warning("generation", "LLM generation failed with cached evidence; returned evidence snippets.")
        trace.metadata["generation_mode"] = "evidence_fallback"
        trace.answer = _format_evidence_fallback(hits, exc)

    try:
        save_trace(trace)
    except Exception as exc:
        logger.warning("Failed to save trace_id=%s: %s", trace.trace_id, exc)

    return {
        "answer": trace.answer,
        "trace_id": trace.trace_id,
    }


def answer(question: str, top_n: int | None = None, collection_name: str | None = None) -> str:
    """Backward-compatible answer API: return only the final answer string."""
    return answer_with_trace(question, top_n=top_n, collection_name=collection_name)["answer"]


if __name__ == "__main__":
    for question in ["等待期内生病了能赔吗？", "酒后驾驶发生事故赔吗？"]:
        print("\n" + "=" * 60)
        print("问题：", question)
        result = answer_with_trace(question)
        print("trace_id：", result["trace_id"])
        print("答案：", result["answer"])
