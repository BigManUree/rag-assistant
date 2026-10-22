"""RAG 问答核心：检索相关片段 -> 拼接上下文 -> 调用 LLM 生成带引用的回答。"""
from src.config import CHAT_MODEL, TOP_K, client
from src.vector_store import query_similar

SYSTEM_PROMPT = """你是一个知识库问答助手。请严格根据下面提供的【参考资料】回答用户问题。
要求：
1. 只能使用参考资料中的信息，不要编造资料中没有的内容。
2. 如果参考资料不足以回答问题，直接说明"根据现有资料无法回答"。
3. 在回答中用 [1] [2] 这样的标注标明信息出处。

【参考资料】
{context}
"""


def _format_context(matches: list[dict]) -> str:
    blocks = []
    for i, m in enumerate(matches, start=1):
        page = f"，第{m['page']}页" if m["page"] else ""
        blocks.append(f"[{i}] 来源：{m['source']}{page}\n{m['text']}")
    return "\n\n".join(blocks)


def answer(question: str, history: list[dict] | None = None):
    """流式生成回答，yield 文本片段；最终通过 .sources 属性可取来源（见 Streamlit 调用）。"""
    matches = query_similar(question, TOP_K)
    if not matches:
        yield "知识库还是空的，请先在左侧上传文档。"
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            context=_format_context(matches))},
    ]
    if history:
        messages.extend(history[-6:])  # 只带最近 3 轮，控制 token
    messages.append({"role": "user", "content": question})

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
        temperature=0.3,  # 问答场景调低温度，减少幻觉
    )
    for chunk in stream:
        # 部分 OpenAI 兼容服务端会发送 choices 为空的心跳/统计块，需跳过
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content

    # 在流末尾附带来源（yield dict 与文本区分）
    yield {"__sources__": matches}
