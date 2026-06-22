# -*- coding: utf-8 -*-
"""使用本地 vLLM 和 BGE-M3 对 InsureRAG 进行 RAGAS 评测。

评测集从外部 JSON（默认 eval/eval_set.json）按文档分组加载，每组指定 collection；
检索与生成都 scope 到该文档的 collection，最终按文档分别出指标并给出总体。
"""

import json
import os
import statistics
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

warnings.filterwarnings("ignore", message=r"Importing .* from 'ragas\.metrics' is deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Langchain.*Wrapper is deprecated.*", category=DeprecationWarning)

import ragas
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config.settings import PROJECT_ROOT, settings
from stage3_search import search
from stage4_generate import answer
from vectorstore.milvus_client import get_milvus_client


VLLM_BASE_URL = settings.llm["base_url"]
VLLM_MODEL = settings.llm["model"]
VLLM_API_KEY = settings.llm["api_key"]
EMBEDDING_MODEL = settings.embedding["model_path"]

EVAL_SET_PATH = Path(os.getenv("INSURERAG_EVAL_SET", str(PROJECT_ROOT / "eval" / "eval_set.json")))
EVAL_RESULT_PATH = PROJECT_ROOT / "eval" / "eval_result.json"  # latest run (overwritten)
EVAL_ARCHIVE_DIR = PROJECT_ROOT / "eval" / "results"           # timestamped archive (kept)
EVAL_LABEL = os.getenv("INSURERAG_EVAL_LABEL", "")             # optional run label, e.g. "newprompt"
# RAGAS judge concurrency. The judge phase (hundreds of LLM calls) dominates
# runtime; vLLM batches concurrent requests, so raising this speeds it up a lot.
EVAL_WORKERS = int(os.getenv("INSURERAG_EVAL_WORKERS", "2"))

METRICS = [answer_correctness, faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = [metric.name for metric in METRICS]


def load_eval_set(path: Path = EVAL_SET_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"评测集文件不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("groups"):
        raise ValueError("评测集缺少 groups。")
    return data


def resolve_local_embedding_model() -> str:
    configured_path = os.getenv("RAGAS_EMBEDDING_MODEL_PATH")
    if configured_path:
        if not os.path.isdir(configured_path):
            raise FileNotFoundError(f"RAGAS_EMBEDDING_MODEL_PATH 指向的目录不存在：{configured_path}")
        return configured_path
    try:
        return snapshot_download(repo_id=EMBEDDING_MODEL, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"本地 Hugging Face 缓存中未找到 {EMBEDDING_MODEL}。请确认模型已下载，"
            "或设置环境变量 RAGAS_EMBEDDING_MODEL_PATH 指向模型目录。"
        ) from exc


def build_ragas_models() -> tuple[Any, Any]:
    judge = ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        max_retries=2,
        timeout=180,
    )
    ragas_llm = LangchainLLMWrapper(judge)

    embedding_path = resolve_local_embedding_model()
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_path,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)
    return ragas_llm, ragas_embeddings


def collect_all_data(eval_set: dict[str, Any]) -> tuple[list[dict[str, Any]], list[float], list[str]]:
    """对每组样本，scope 到该文档的 collection 检索 + 生成，收集 RAGAS 行。"""
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    documents: list[str] = []
    final_top_k = int(settings.retrieval["final_top_k"])

    for group in eval_set["groups"]:
        document = group["document"]
        collection = group["collection"]
        samples = group.get("samples", [])
        print(f"\n===== 文档：{document}  (collection={collection}, {len(samples)} 题) =====")

        # Guard against a stale/wrong collection silently scoring 0 recall.
        try:
            if not get_milvus_client().has_collection(collection):
                print(f"  ⚠️  collection 「{collection}」 不存在，跳过该文档的 {len(samples)} 题。")
                print("      请确认该文档已入库，或在 eval_set.json 中把 collection 改成实际值。")
                continue
        except Exception as exc:
            print(f"  ⚠️  无法确认 collection 「{collection}」 是否存在（{exc.__class__.__name__}），仍尝试评测。")

        for index, sample in enumerate(samples, start=1):
            question = sample["question"]
            started_at = time.perf_counter()
            try:
                hits = search(question, top_n=final_top_k, collection_name=collection)
                contexts = [hit.get("parent_text", "") for hit in hits]
                response = answer(question, collection_name=collection)
            except Exception as exc:
                print(f"  [{index}/{len(samples)}] 检索/生成失败：{exc}")
                contexts, response = [], "(检索或生成失败)"
            latencies.append(time.perf_counter() - started_at)

            rows.append({
                "user_input": question,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": sample["ground_truth"],
            })
            documents.append(document)
            print(f"  [{index}/{len(samples)}] {question}  -> 召回 {len(contexts)} 条")

    return rows, latencies, documents


def _metric_means(frame: Any) -> dict[str, float]:
    means = {}
    for name in METRIC_NAMES:
        if name in frame:
            value = frame[name].mean()
            means[name] = round(float(value), 4) if value == value else 0.0  # NaN-safe
    return means


def summarize(result: Any, documents: list[str], latencies: list[float]) -> dict[str, Any]:
    frame = result.to_pandas()
    frame["document"] = documents
    frame["latency_s"] = [round(x, 3) for x in latencies]

    print("\n" + "=" * 72)
    print("总体指标")
    print("=" * 72)
    overall = _metric_means(frame)
    for name, value in overall.items():
        print(f"{name:<20}: {value:.4f}")

    per_document: dict[str, Any] = {}
    print("\n按文档指标")
    for document, sub in frame.groupby("document"):
        means = _metric_means(sub)
        per_document[document] = {"count": int(len(sub)), **means}
        print(f"\n--- {document}  ({len(sub)} 题) ---")
        for name, value in means.items():
            print(f"  {name:<20}: {value:.4f}")

    print("\n响应延迟")
    print(f"  平均：{statistics.mean(latencies):.2f}s  最快：{min(latencies):.2f}s  最慢：{max(latencies):.2f}s")

    # 每题明细（耗时 + 各指标得分），并打印耗时最长的几道
    def _cell(value: Any) -> Any:
        return None if value != value else round(float(value), 4)  # NaN-safe

    samples: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record = {
            "document": row["document"],
            "question": str(row["user_input"]),
            "latency_s": float(row["latency_s"]),
        }
        for name in METRIC_NAMES:
            if name in frame:
                record[name] = _cell(row[name])
        samples.append(record)

    print("\n耗时最长的问题 (Top 5)")
    for record in sorted(samples, key=lambda r: r["latency_s"], reverse=True)[:5]:
        print(f"  {record['latency_s']:>6.2f}s  [{record['document']}] {record['question'][:38]}")

    return {
        "model": VLLM_MODEL,
        "enable_thinking": bool(settings.llm["enable_thinking"]),
        "label": EVAL_LABEL,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_samples": int(len(frame)),
        "overall": overall,
        "per_document": per_document,
        "latency": {
            "avg_s": round(statistics.mean(latencies), 3),
            "min_s": round(min(latencies), 3),
            "max_s": round(max(latencies), 3),
        },
        "samples": samples,
    }


def main() -> None:
    print(f"RAGAS 版本：{ragas.__version__}")
    print(f"评测集：{EVAL_SET_PATH}")
    print(f"裁判模型：{VLLM_MODEL} ({VLLM_BASE_URL})")
    print(f"判分并发 max_workers={EVAL_WORKERS}（可用环境变量 INSURERAG_EVAL_WORKERS 调整）")

    eval_set = load_eval_set()
    rows, latencies, documents = collect_all_data(eval_set)

    evaluation_dataset = EvaluationDataset.from_list(rows)
    ragas_llm, ragas_embeddings = build_ragas_models()
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(max_workers=EVAL_WORKERS, timeout=300, max_retries=3),
        raise_exceptions=False,
        show_progress=True,
    )

    summary = summarize(result, documents, latencies)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)

    # "latest" pointer (overwritten each run)
    EVAL_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULT_PATH.write_text(payload, encoding="utf-8")

    # timestamped archive (never overwritten, for A/B comparison)
    EVAL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"_{EVAL_LABEL}" if EVAL_LABEL else ""
    archive_path = EVAL_ARCHIVE_DIR / f"eval_{stamp}{label}.json"
    archive_path.write_text(payload, encoding="utf-8")

    print(f"\n结果已保存到：{EVAL_RESULT_PATH}")
    print(f"存档副本：{archive_path}")


if __name__ == "__main__":
    main()
