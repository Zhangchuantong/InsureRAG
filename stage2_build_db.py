# -*- coding: utf-8 -*-
"""
阶段 2：embedding + 入库 Milvus Lite
================================================
对应简历技术点：
  - "使用 bge-m3 对文档同时生成稠密向量与稀疏向量，分别存储，支持混合检索"
  - "设计多集合存储策略"（这里先做单集合，多集合在注释里说明怎么扩展）

核心思路（面试要点）：
  bge-m3 的杀手锏：一个模型同时输出
    - dense（稠密向量）：捕捉语义相似（"能赔吗"≈"是否承担给付责任"）
    - sparse（稀疏向量，类似 BM25）：捕捉关键词精确匹配（"等待期""既往症"这种术语）
  保险问答里两者缺一不可：术语必须精确命中，口语化又必须靠语义。
  所以把两种向量都存进 Milvus，检索时各查一路再融合（见 stage3）。

  为什么用 Milvus 而不是 FAISS：
    - 原生支持稀疏向量 + 稠密向量的混合检索（hybrid_search）
    - 支持多集合（collection），天然适合按险种隔离
    - Milvus Lite 本地零部署，开发期就能用，上线可平滑切到 Milvus 集群
"""

import os
import json
import pickle

from pymilvus import (
    MilvusClient, DataType, Function, FunctionType,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
DB_PATH = os.path.join(DATA_DIR, "insure_rag.db")   # Milvus Lite 数据文件
CACHE_PATH = os.path.join(DATA_DIR, "retrieval_cache.pkl")
COLLECTION = "policy_clauses"                         # 集合名（多险种时可建多个）


def get_embedding_model():
    """加载 bge-m3。首次会自动下载模型（约 2GB），耐心等。
    use_fp16=False 兼容 CPU；有 GPU 可设 device='cuda', use_fp16=True 提速。
    """
    print("【阶段2】加载 bge-m3（首次需下载模型）…")
    return BGEM3EmbeddingFunction(use_fp16=False, device="cpu")


def build_collection(client: MilvusClient, dense_dim: int):
    """建集合：定义字段(field)与索引(index)。

    字段设计就是简历里"结合数据特点设计 field 和 index"：
      - id           主键
      - child_text   子块原文（检索用）
      - parent_text  父块原文（喂大模型用）
      - source       来源，可做过滤
      - dense_vector 稠密向量
      - sparse_vector 稀疏向量
    """
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)  # 重跑时清空，方便调试

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("child_text", DataType.VARCHAR, max_length=2000)
    schema.add_field("parent_text", DataType.VARCHAR, max_length=8000)
    schema.add_field("source", DataType.VARCHAR, max_length=256)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dense_dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    # 索引：稠密用 IP（内积，bge 向量已归一化，IP≈cosine）；稀疏用专用索引
    # index_params = client.prepare_index_params()
    # index_params.add_index(
    #     field_name="dense_vector",
    #     index_type="AUTOINDEX",
    #     metric_type="IP",
    # )
    # index_params.add_index(
    #     field_name="sparse_vector",
    #     index_type="SPARSE_INVERTED_INDEX",
    #     metric_type="IP",
    # )

    # client.create_collection(
    #     collection_name=COLLECTION,
    #     schema=schema,
    #     index_params=index_params,
    # )
    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
    )
    print(f"  集合 {COLLECTION} 创建完成（稠密维度={dense_dim}）")


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"【阶段2】读取 {len(chunks)} 个子块")

    bge = get_embedding_model()

    # 一次性对所有子块编码，拿到稠密+稀疏两套向量
    texts = [c["child_text"] for c in chunks]
    print("【阶段2】编码中（同时产出 dense + sparse）…")
    embeddings = bge(texts)   # dict: {"dense": ndarray, "sparse": csr_matrix}
    dense_vecs = embeddings["dense"]
    sparse_vecs = embeddings["sparse"]
    dense_dim = len(dense_vecs[0])

    client = MilvusClient(uri=DB_PATH)
    build_collection(client, dense_dim)

    # 组装入库数据
    rows = []
    for i, c in enumerate(chunks):
        rows.append({
            "id": c["id"],
            "child_text": c["child_text"],
            "parent_text": c["parent_text"],
            "source": c["source"],
            "dense_vector": dense_vecs[i],
            "sparse_vector": sparse_vecs[[i]],   # 取第 i 行稀疏向量
        })

    client.insert(collection_name=COLLECTION, data=rows)
    client.flush(COLLECTION)
    print(f"  已插入 {len(rows)} 条数据到 Milvus Lite（{DB_PATH}）")
    print("  阶段2 完成，可运行 stage3_search.py 测试检索。")

    # ───── 多集合扩展说明（面试可讲）─────
    # 真实项目按险种隔离：重疾→collection_critical，医疗→collection_medical …
    # 好处：1) 检索时按用户意图只查对应集合，减少干扰、提速
    #       2) 不同险种字段/更新频率不同，分开管理更清晰
    # 实现上就是把 COLLECTION 参数化，对每个险种各建一个集合即可。


if __name__ == "__main__":
    main()
