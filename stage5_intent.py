# -*- coding: utf-8 -*-
"""
阶段 5：意图识别 + Query 改写（完整入口）
================================================
对应简历技术点：
  - "基于 BERT/规则训练意图识别模块，支持 4 种意图分类"
  - "设计 query 改写机制，将口语化提问转为检索友好查询，提升召回率"

为什么需要意图识别（面试要点）：
  不是所有问题都该走 RAG。"你好""谢谢"这种闲聊走 RAG 纯属浪费且会瞎答；
  不同业务意图（条款/理赔/产品咨询）未来可路由到不同集合或不同 Prompt。
  意图识别就是问答系统的"分流器"。

为什么需要 query 改写：
  用户问"我这个能赔吗"信息量极低，直接检索召回很差。
  改写成"保险责任 赔付条件 责任免除"这类含关键术语的查询，召回率显著提升。
  这里用大模型做改写（zero-shot），简历也可写"基于规则/小模型"，看你实现哪种。

说明：
  这里的意图识别用「大模型 zero-shot 分类」做演示（最快跑通）。
  简历若写"BERT 训练意图识别"，可另用少量标注数据微调 bert-base-chinese，
  原理一样（文本→4分类），这里给规则+大模型版本让你先看到效果。
"""

import json
import urllib.request

from stage4_generate import call_vllm, answer

# 4 种意图，对应我们之前定的分流
INTENTS = ["条款解释", "理赔流程", "产品咨询", "闲聊"]

INTENT_PROMPT = """请判断用户问题属于以下哪一类意图，只回答类别名称，不要其他内容。

分类标准：
- 条款解释：询问能不能赔、保不保、等待期、免责、既往症、责任范围等。
- 理赔流程：询问怎么申请理赔、需要什么材料、多久到账、手续流程等。
- 产品咨询：询问保险产品适合谁、怎么买、多少钱、保障内容对比等。
- 闲聊：你好、谢谢、你是谁等非保险业务问题。

类别：条款解释 / 理赔流程 / 产品咨询 / 闲聊

用户问题：{question}
意图类别："""

REWRITE_PROMPT = """用户的保险问题口语化、信息不全。请将其改写为一句包含关键保险术语、便于检索的查询。
只输出改写后的查询，不要解释。

原问题：{question}
改写后："""




def classify_intent(question: str) -> str:
    """意图识别：返回 4 类之一。"""
    resp = call_vllm(INTENT_PROMPT.format(question=question))
    # 容错：取模型输出里第一个命中的意图词
    for intent in INTENTS:
        if intent in resp:
            return intent
    return "条款解释"   # 兜底默认走条款解释


def rewrite_query(question: str) -> str:
    """query 改写：口语化 → 检索友好。"""
    rewritten = call_vllm(REWRITE_PROMPT.format(question=question))
    return rewritten.strip().strip('"。')


def chat(question: str) -> str:
    """完整入口：意图识别 → 分流 → (改写 → 检索 → 生成)。"""
    intent = classify_intent(question)
    print(f"  [意图识别] → {intent}")

    if intent == "闲聊":
        # 闲聊不走 RAG，直接兜底回复
        return "您好，我是保险条款问答助手，可以帮您解答保险条款、理赔流程等问题，请问有什么可以帮您？"

    # 业务意图：先改写再检索生成
    rewritten = rewrite_query(question)
    print(f"  [Query改写] {question} → {rewritten}")

    # 用改写后的 query 去检索生成（原问题也可一并传给生成保留语气，这里简化）
    return answer(rewritten)


if __name__ == "__main__":
    questions = [
        "你好呀",                    # 闲聊 → 兜底
        "我刚买没多久就生病了能赔吗",   # 条款解释（等待期）
        "出险了我该怎么弄手续",         # 理赔流程
    ]
    for q in questions:
        print("\n" + "=" * 60)
        print("用户：", q)
        print("助手：", chat(q))
