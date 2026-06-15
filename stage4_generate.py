# -*- coding: utf-8 -*-

from openai import OpenAI

from stage3_search import search

BASE_URL = "http://localhost:8002/v1"
API_KEY = "EMPTY"
MODEL = "Qwen/Qwen3-8B-AWQ"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

PROMPT_TEMPLATE = """你是一名专业、严谨的保险条款解答助手。请依据下面的【参考条款】回答用户问题。

要求:
1. 以【参考条款】为唯一依据。条款里已写明、或可由条款内容直接得出的信息,就要据此明确回答——即使条款表述和用户问法不一样,只要条款能回答,就不要因为字面不同而拒答。
2. 【免责情形必须明确说"不赔"】:如果【参考条款】把用户所问的情形列入"责任免除""不负保险责任""不承担给付保险金责任"等,要明确回答该情形保险公司不予赔付并说明依据,绝不能因为是否定情形就回答"无法确定"。
   (例如:条款将"战争"列入责任免除,用户问"战争中身故赔吗",应明确回答"不赔,战争属于责任免除情形"。)
3. 不得编造条款中没有的事实、数字或条件。
4. 只有当【参考条款】里确实找不到任何与问题相关的内容时,才回答:"根据现有条款,无法确定该问题,建议联系保险公司客服确认。"
5. 回答清晰、口语化;关键数字、期限、条件保留条款原文(如"六十日""基本保额的二倍""一百八十日")。
6. 末尾用一行标注依据来源(如:依据:第五条)。

【参考条款】
{context}

【用户问题】
{question}

【你的回答】"""


def call_vllm(prompt: str) -> str:
    """调用本地 vLLM OpenAI-Compatible API。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一名严谨的保险条款解答助手，只能依据给定条款回答。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=512,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
    )
    return resp.choices[0].message.content.strip()


def answer(question: str, top_n: int = 3) -> str:
    hits = search(question, top_n=top_n)

    context_parts = []
    for h in hits:
        context_parts.append(f"（来源：{h['source']}）\n{h['parent_text']}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    return call_vllm(prompt)


if __name__ == "__main__":
    questions = [
        "等待期内生病了能赔吗？",
        "酒驾出了事故保险公司会赔吗？",
        "申请理赔需要准备什么材料？",
    ]

    for q in questions:
        print("\n" + "=" * 60)
        print("问：", q)
        print("答：", answer(q))
