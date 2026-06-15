# -*- coding: utf-8 -*-
"""
阶段 3：混合检索 + 重排序（项目核心，面试必问）
================================================
对应简历技术点：
  - "使用稠密+稀疏混合检索"
  - "使用 bge-reranker 对召回结果重排，提升 top-k 准确性"

两段式检索的原因（务必讲清楚）：
  1) 召回(retrieve)：要"宁可错召、不可漏召"，所以用混合检索拉回较多候选(如 top20)。
     - 稠密一路 + 稀疏一路，分别召回，再用 RRF 融合排名。
     - 为什么用 RRF 而不是分数直接相加？因为稠密分数(IP)和稀疏分数(BM25)
       量纲不同、不可比；RRF 只看"名次"，天然解决量纲问题，是工业界常用做法。
  2) 重排(rerank)：召回是"双塔"粗排（query 和 doc 分开编码，快但不够准）。
     reranker 是"交叉编码器"，把 query 和 doc 拼一起喂模型打分，更准但慢，
     所以只对粗排 top20 精排，取最终 top3。这就是"先广后精"的经典两段式。
"""

import os

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from sentence_transformers import CrossEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "insure_rag.db")
COLLECTION = "policy_clauses"

_bge = None
_reranker = None


def get_bge():
    global _bge
    if _bge is None:
        _bge = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    return _bge


def get_reranker():
    """bge-reranker-base：交叉编码器，对 (query, doc) 对打分。"""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    return _reranker


def hybrid_retrieve(client, query: str, top_k: int = 20):
    """混合检索：稠密 + 稀疏两路召回，RRF 融合。"""
    bge = get_bge()
    q_emb = bge([query])
    q_dense = q_emb["dense"][0]
    q_sparse = q_emb["sparse"][[0]]

    # 稠密一路
    dense_req = AnnSearchRequest(
        data=[q_dense], anns_field="dense_vector",
        param={"metric_type": "IP"}, limit=top_k,
    )
    # 稀疏一路
    sparse_req = AnnSearchRequest(
        data=[q_sparse], anns_field="sparse_vector",
        param={"metric_type": "IP"}, limit=top_k,
    )

    # RRF 融合两路排名
    results = client.hybrid_search(
        collection_name=COLLECTION,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),     # k 是 RRF 平滑常数，60 是常用默认
        limit=top_k,
        output_fields=["child_text", "parent_text", "source"],
    )
    return results[0]   # 单 query，取第一组


def rerank(query: str, candidates, top_n: int = 3):
    """对粗排候选用 reranker 精排，返回 top_n。"""
    reranker = get_reranker()

    pairs = [[query, c["entity"]["child_text"]] for c in candidates]

    scores = reranker.predict(pairs)
    scores = [float(s) for s in scores]

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def search(query: str, top_n: int = 3):
    """对外统一入口：混合检索 → 重排 → 返回 top_n（含父块上下文）。"""
    client = MilvusClient(uri=DB_PATH)
    client.load_collection(COLLECTION)
    candidates = hybrid_retrieve(client, query, top_k=20)

    ranked = rerank(query, candidates, top_n=top_n)

    out = []
    for cand, score in ranked:
        out.append({
            "score": float(score),
            "child_text": cand["entity"]["child_text"],
            "parent_text": cand["entity"]["parent_text"],   # 喂大模型用父块
            "source": cand["entity"]["source"],
        })
    return out


if __name__ == "__main__":
    test_queries = [
        "等待期是多久，期间生病能赔吗",
        "喝酒开车出事故保险赔不赔",
        "理赔要准备哪些材料",
    ]
    for q in test_queries:
        print("\n" + "=" * 50)
        print("问题：", q)
        results = search(q, top_n=2)
        for r in results:
            print(f"  [score={r['score']:.3f}] {r['child_text'][:50]}…")
