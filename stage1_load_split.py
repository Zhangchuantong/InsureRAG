# -*- coding: utf-8 -*-
"""
阶段 1：文档加载 + 分块
================================================
对应简历技术点：
  - "自定义 PDF/Word 加载器，实现对保险条款等文件的解析读取"
  - "利用条款章节结构，使用 LangChain 切分器构建父子分块"

核心思路（面试要点）：
  保险条款是结构化长文档（第X条…），直接按固定长度硬切会割裂语义。
  所以采用「父子分块」：
    - 子块(child)：粒度小，用于精确检索（向量召回更准）
    - 父块(parent)：粒度大，命中子块后回溯父块，给大模型更完整的上下文
  这是 RAG 里很常见的"小块检索、大块喂模型"策略，能兼顾召回准确性和上下文完整性。
"""

import os
import re
import json


from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "chunks.json")

# 没有 PDF 时的兜底示例文本（让你能立刻跑通闭环）
SAMPLE_TEXT = """第一条 保险合同的构成
本保险合同由保险条款、投保单、保险单、批单以及其他相关投保资料共同构成。

第二条 投保范围
凡年龄在出生满28天至65周岁、身体健康者，均可作为被保险人投保本保险。

第三条 保险责任 等待期
本合同等待期为90天。被保险人在等待期内因疾病导致的相关情况，本公司不承担给付保险金的责任，但退还已交保险费。等待期内因意外伤害导致的，不受等待期限制。

第四条 责任免除
因下列情形之一导致被保险人发生保险事故的，本公司不承担给付保险金的责任：
（一）投保人对被保险人的故意杀害、故意伤害；
（二）被保险人故意自伤、主动吸食或注射毒品；
（三）被保险人酒后驾驶、无合法有效驾驶证驾驶。

第五条 保险金申请与给付 理赔流程
保险事故发生后，受益人应在10日内通知本公司，并提供以下材料：保险合同、被保险人身份证明、医疗诊断证明、医疗费用原始凭证等。本公司在收齐材料后5日内作出核定。

第六条 既往症约定
既往症是指被保险人在本合同生效前已患或已有症状的疾病。对于既往症导致的相关费用，本公司不承担给付责任。
"""


def load_pdfs(data_dir: str):
    """加载 data 目录下所有 PDF。这一步就是简历说的'自定义文档加载器'雏形。

    返回：[(来源文件名, 全文文本), ...]
    """
    docs = []
    if not os.path.isdir(data_dir):
        return docs

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return docs

    # 延迟导入，避免没装 pypdf 时整个文件跑不起来
    from langchain_community.document_loaders import PyPDFLoader

    for fname in pdf_files:
        path = os.path.join(data_dir, fname)
        loader = PyPDFLoader(path)
        pages = loader.load()  # 每页一个 Document
        full_text = "\n".join(p.page_content for p in pages)
        docs.append((fname, full_text))
        print(f"  已加载 PDF：{fname}（{len(pages)} 页）")
    return docs


def clean_repeated_pdf_text(text: str) -> str:
    """清理部分老 PDF 文本层中连续重复 3~4 次的标题和短语。"""
    cleaned_lines = []
    for line in text.splitlines():
        previous = None
        while previous != line:
            previous = line
            line = re.sub(r"(.{2,40}?)\1{2,3}", r"\1", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def split_by_article(text: str):
    """按"第X条"把条款切成父块。

    保险条款有天然的"第一条/第二条…"结构，这是最自然的父块边界，
    比按字数硬切语义完整得多 —— 这正是简历里"利用条款章节结构"的含义。
    """
    text = clean_repeated_pdf_text(text)

    # 用正则在"第X条"前面切分（保留分隔符）
    parts = re.split(r"(?=第[一二三四五六七八九十百零〇两\d]+条)", text)
    parents = [p.strip() for p in parts if p.strip()]
    # 如果文本没有"第X条"结构（比如普通产品手册），退化为整段作为一个父块
    if not parents:
        parents = [text.strip()]
    return parents


def build_chunks(docs):
    """父子分块：父块=条款，子块=父块内再细切。"""
    # 子块切分器：粒度小，便于精确向量召回
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,        # 子块约 200 字
        chunk_overlap=30,      # 重叠 30 字，避免边界语义断裂
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    chunks = []
    chunk_id = 0
    for source, text in docs:
        parents = split_by_article(text)
        for p_idx, parent_text in enumerate(parents):
            children = child_splitter.split_text(parent_text)
            for child_text in children:
                chunks.append({
                    "id": chunk_id,
                    "source": source,        # 来源文件，可用于多集合/过滤
                    "parent_id": p_idx,      # 子块指向的父块
                    "parent_text": parent_text,   # 喂给大模型的完整上下文
                    "child_text": child_text,     # 真正拿去做向量检索的文本
                })
                chunk_id += 1
    return chunks


def main():
    print("【阶段1】加载文档…")
    docs = load_pdfs(DATA_DIR)

    if not docs:
        print("  data/ 下没有 PDF，使用内置示例文本（先跑通闭环）。")
        docs = [("sample_policy.txt", SAMPLE_TEXT)]

    print("【阶段1】父子分块…")
    chunks = build_chunks(docs)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    n_parents = len({(c["source"], c["parent_id"]) for c in chunks})
    print(f"  完成：{len(chunks)} 个子块，{n_parents} 个父块 → 已写入 {OUTPUT_PATH}")
    print("  示例子块：", chunks[0]["child_text"][:40], "…")


if __name__ == "__main__":
    main()
