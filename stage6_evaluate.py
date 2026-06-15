# -*- coding: utf-8 -*-
"""使用本地 vLLM 和 BGE-M3 对 InsureRAG 进行 RAGAS 评测。"""

import os
import statistics
import time
import warnings
from typing import Any

# 强制 Hugging Face 组件仅使用本地缓存，避免评测过程中意外联网。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 当前 RAGAS 0.4.3 仍支持任务要求的 LangChain Wrapper，只是会输出弃用提示。
warnings.filterwarnings(
    "ignore",
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Langchain.*Wrapper is deprecated.*",
    category=DeprecationWarning,
)

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

from stage3_search import search
from stage4_generate import answer


VLLM_BASE_URL = "http://localhost:8002/v1"
VLLM_MODEL = "Qwen/Qwen3-8B-AWQ"
VLLM_API_KEY = "EMPTY"
EMBEDDING_MODEL = "BAAI/bge-m3"

# 评测集依据 data/保险条款.pdf 中的《康宁终身保险条款》编写。
EVALUATION_SAMPLES: list[dict[str, str]] = [
    {
        "question": "投保康宁终身保险有年龄限制吗?",
        "ground_truth": "有。凡七十周岁以下、身体健康者,均可作为被保险人投保。",
    },  # 第二条 投保范围
    {
        "question": "确诊重大疾病能赔多少?有没有时间要求?",
        "ground_truth": "被保险人在合同生效(或复效)之日起一百八十日后初次发生、并经二级以上"
                        "(含二级)医院确诊患重大疾病的,公司按基本保额的二倍给付重大疾病保险金。",
    },  # 第四条 保险责任 一
    {
        "question": "被保险人身故能赔多少?",
        "ground_truth": "公司按基本保额的三倍给付身故保险金,但应扣除已给付的重大疾病保险金,本合同终止。",
    },  # 第四条 保险责任 二
    {
        "question": "被保险人身体高度残疾赔多少?",
        "ground_truth": "公司按基本保额的三倍给付高度残疾保险金,但应扣除已给付的重大疾病保险金,本合同终止。",
    },  # 第四条 保险责任 三
    {
        "question": "拿了重大疾病保险金之后,还要继续交保费吗?",
        "ground_truth": "若重大疾病保险金的给付发生于交费期内,从给付之日起,免交以后各期保险费,本合同继续有效。",
    },  # 第四条 保险责任 一
    {
        "question": "喝酒开车出了事故,保险公司赔吗?",
        "ground_truth": "不赔。被保险人酒后驾驶、无有效驾驶执照驾驶,或驾驶无有效行驶证的机动交通工具,"
                        "导致身故、高度残疾或患重大疾病的,公司不负保险责任。",
    },  # 第五条 责任免除 五
    {
        "question": "被保险人自杀,保险公司赔吗?",
        "ground_truth": "被保险人在本合同生效(或复效)之日起二年内自杀的,公司不负保险责任。",
    },  # 第五条 责任免除 四
    {
        "question": "我刚买没多久就查出大病,能赔吗?",
        "ground_truth": "不赔。被保险人在本合同生效(或复效)之日起一百八十日内患重大疾病,"
                        "或因疾病身故或造成身体高度残疾的,公司不负保险责任。",
    },  # 第五条 责任免除 七
    {
        "question": "因为吸毒出的事,保险赔吗?",
        "ground_truth": "不赔。被保险人服用、吸食或注射毒品,导致身故、高度残疾或患重大疾病的,公司不负保险责任。",
    },  # 第五条 责任免除 三
    {
        "question": "康宁终身保险的保险费可以怎么交?",
        "ground_truth": "保险费交付方式分为趸交、年交、半年交;分期交付保险费的交费期间分为十年和二十年,"
                        "由投保人在投保时选择。",
    },  # 第六条 保险费
    {
        "question": "宽限期是多久?宽限期内发生保险事故还承担责任吗?",
        "ground_truth": "未按期交付保险费的,自次日起六十日为宽限期;在宽限期内发生保险事故,公司仍负保险责任。",
    },  # 第七条 宽限期间
    {
        "question": "过了宽限期还没交保费,合同会怎么样?",
        "ground_truth": "逾宽限期仍未交付保险费的,如合同现金价值扣除欠交保险费及利息、借款及利息后的余额"
                        "足以垫交到期应交保险费,公司将自动垫交使合同继续有效;当现金价值余额不足以垫交,"
                        "或垫交的保险费及利息达到现金价值时,本合同效力中止。",
    },  # 第七条 保险费自动垫交及合同效力中止
    {
        "question": "保险合同效力中止后还能恢复吗?需要办什么?",
        "ground_truth": "可以。在合同效力中止之日起二年内,投保人可填写复效申请书,并提供被保险人的健康声明书"
                        "或二级以上(含二级)医院出具的体检报告书,申请恢复合同效力;经公司审核同意,"
                        "自投保人补交所欠保险费及利息的次日起,本合同效力恢复。",
    },  # 第八条 合同效力恢复
    {
        "question": "保费交不起又不想退保,能不能减少保额把合同保留下来?",
        "ground_truth": "可以。在合同具有现金价值的情况下,投保人可以按合同当时的现金价值在扣除欠交保险费及利息、"
                        "借款及利息后的余额,作为一次交清的全部保险费,以相同的合同条件减少保险金额,"
                        "本合同继续有效(此项选择不适用于次标准体)。",
    },  # 第九条 减额交清保险
    {
        "question": "我想更换保险受益人,要怎么办?",
        "ground_truth": "被保险人或投保人可以变更受益人,但需书面通知公司,经公司在保险单上批注后方能生效;"
                        "投保人指定或变更受益人时须经被保险人书面同意。",
    },  # 第十一条 受益人的指定和变更
    {
        "question": "出险之后多久要通知保险公司?",
        "ground_truth": "投保人、被保险人或受益人应于知悉保险事故发生之日起十日内以书面形式通知公司;"
                        "否则应承担由于通知迟延致使公司增加的查勘、调查费用,但因不可抗力导致迟延的除外。",
    },  # 第十三条 保险事故通知
    {
        "question": "申请重大疾病保险金要准备哪些材料?",
        "ground_truth": "由被保险人或其委托的代理人作为申请人,填写保险金给付申请书,并提交:保险合同及"
                        "最近一次保险费的交费凭证;被保险人的户籍证明与身份证件;二级以上(含二级)医院出具的"
                        "疾病诊断证明书;如为代理人,还应提供授权委托书、身份证明等相关资料。",
    },  # 第十四条 保险金申请 一
    {
        "question": "保险公司多久内给付保险金?",
        "ground_truth": "公司收到申请书及相关证明、资料后,对核定属于保险责任的,在与申请人达成给付协议后"
                        "十日内履行给付义务;对不属于保险责任的,向申请人发出拒绝给付保险金通知书。",
    },  # 第十四条 保险金申请 四
    {
        "question": "申请理赔有时间限制吗?",
        "ground_truth": "被保险人或受益人对公司请求给付保险金的权利,自其知道保险事故发生之日起五年不行使而消灭。",
    },  # 第十四条 保险金申请 五
    {
        "question": "可以用这份保单借款吗?最多能借多少?",
        "ground_truth": "在合同已具有现金价值时,投保人可以书面形式申请借款,最高借款金额不得超过合同当时的"
                        "现金价值在扣除欠交保险费及利息、借款及利息后余额的百分之七十,每次借款时间不得超过六个月。",
    },  # 第十五条 借款
    {
        "question": "刚买就后悔了,退保能拿回多少钱?",
        "ground_truth": "投保人于签收保险单后十日内要求解除合同的,公司退还已收全部保险费,"
                        "但如经公司体检的,则应扣除体检费。",
    },  # 第二十一条 投保人解除合同的处理
    {
        "question": "这份保险保哪些重大疾病?",
        "ground_truth": "重大疾病指下列疾病或手术之一:心脏病(心肌梗塞)、冠状动脉旁路手术、脑中风后遗症、"
                        "慢性肾衰竭(尿毒症)、癌症、瘫痪、重大器官移植手术、严重烧伤、暴发性肝炎、主动脉手术。",
    }  # 第二十三条 释义(重大疾病)
]

METRICS = [
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]


def resolve_local_embedding_model() -> str:
    """从 Hugging Face 缓存解析模型目录，不触发网络下载。"""
    configured_path = os.getenv("RAGAS_EMBEDDING_MODEL_PATH")
    if configured_path:
        if not os.path.isdir(configured_path):
            raise FileNotFoundError(
                "RAGAS_EMBEDDING_MODEL_PATH 指向的目录不存在："
                f"{configured_path}"
            )
        return configured_path

    try:
        return snapshot_download(
            repo_id=EMBEDDING_MODEL,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"本地 Hugging Face 缓存中未找到 {EMBEDDING_MODEL}。"
            "请确认模型已下载，或设置环境变量 "
            "RAGAS_EMBEDDING_MODEL_PATH 指向模型目录。"
        ) from exc


def build_ragas_models() -> tuple[Any, Any]:
    """构造只访问本地服务和本地模型文件的 RAGAS 模型包装器。"""
    judge = ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
        max_retries=2,
        timeout=180,
    )
    ragas_llm = LangchainLLMWrapper(judge)

    # 使用 snapshot 绝对路径可避免 transformers 再次访问 Hugging Face。
    embedding_path = resolve_local_embedding_model()
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_path,
        model_kwargs={
            "device": "cpu",
            "local_files_only": True,
        },
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)
    return ragas_llm, ragas_embeddings


def collect_evaluation_data() -> tuple[list[dict[str, Any]], list[float]]:
    """复用项目现有检索和生成接口，收集 RAGAS 样本及端到端延迟。"""
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    for index, sample in enumerate(EVALUATION_SAMPLES, start=1):
        question = sample["question"]
        print(f"\n[{index}/{len(EVALUATION_SAMPLES)}] {question}")

        started_at = time.perf_counter()
        hits = search(question, top_n=5)
        contexts = [hit["parent_text"] for hit in hits]
        generated_answer = answer(question)
        elapsed = time.perf_counter() - started_at

        rows.append(
            {
                "user_input": question,
                "response": generated_answer,
                "retrieved_contexts": contexts,
                "reference": sample["ground_truth"],
            }
        )
        latencies.append(elapsed)

        print(f"  召回上下文：{len(contexts)} 条")
        print(f"  生成答案：{generated_answer}")
        print(f"  端到端延迟：{elapsed:.2f} 秒")

    return rows, latencies


def print_results(result: Any, latencies: list[float]) -> None:
    """打印总体指标、逐条指标和延迟统计。"""
    metric_names = [metric.name for metric in METRICS]
    result_frame = result.to_pandas()

    print("\n" + "=" * 72)
    print("RAGAS 总体指标")
    print("=" * 72)
    for name in metric_names:
        valid_count = result_frame[name].notna().sum()
        total_count = len(result_frame)
        print(
            f"{name:<20}: {result_frame[name].mean():.4f} "
            f"（有效样本 {valid_count}/{total_count}）"
        )

    print("\n逐条指标")
    display_columns = ["user_input", *metric_names]
    print(result_frame[display_columns].to_string(index=False))

    print("\n响应延迟")
    print(f"平均延迟：{statistics.mean(latencies):.2f} 秒")
    print(f"最快延迟：{min(latencies):.2f} 秒")
    print(f"最慢延迟：{max(latencies):.2f} 秒")


def main() -> None:
    print(f"RAGAS 版本：{ragas.__version__}")
    print(f"裁判模型：{VLLM_MODEL} ({VLLM_BASE_URL})")
    print(f"Embedding：{EMBEDDING_MODEL}（仅使用本地缓存）")

    rows, latencies = collect_evaluation_data()
    evaluation_dataset = EvaluationDataset.from_list(rows)
    ragas_llm, ragas_embeddings = build_ragas_models()

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            max_workers=1,
            timeout=180,
            max_retries=3,
        ),
        # 本地小模型偶尔可能输出不符合 RAGAS schema 的内容。单项记为 NaN，
        # 保留其余评测结果，避免长时间运行因一个样本全部作废。
        raise_exceptions=False,
        show_progress=True,
    )
    print_results(result, latencies)


if __name__ == "__main__":
    main()
