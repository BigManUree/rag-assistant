"""Streamlit 界面：上传文档、基于知识库对话、展示引用来源。

运行： streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

from src.config import DATA_DIR, ZHIPU_API_KEY
from src.document_loader import split_into_chunks
from src.vector_store import add_chunks, get_collection
from src.rag import answer

st.set_page_config(page_title="RAG 知识库问答", page_icon="📚")
st.title("📚 RAG 知识库问答助手")

if not ZHIPU_API_KEY or ZHIPU_API_KEY == "your_key_here":
    st.error("未检测到 LLM_API_KEY，请先在项目根目录 .env 文件中配置后重启。")
    st.stop()

# --- 侧边栏：上传与知识库状态 ---
with st.sidebar:
    st.header("知识库管理")
    uploaded = st.file_uploader(
        "上传文档（PDF / TXT / MD）",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("导入知识库", type="primary"):
        progress = st.progress(0, text="正在解析并向量化...")
        for i, f in enumerate(uploaded):
            save_path = DATA_DIR / f.name
            save_path.write_bytes(f.getbuffer())
            count = add_chunks(split_into_chunks(save_path))
            st.write(f"✅ {f.name}：{count} 个文本块")
            progress.progress((i + 1) / len(uploaded))
        st.success("导入完成")

    try:
        doc_count = get_collection().count()
    except Exception:
        doc_count = 0
    st.caption(f"当前知识库：{doc_count} 个文本块")

# --- 对话区 ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content"}]，只存对话不存来源

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("针对知识库内容提问吧"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[:-1]]
        full = ""
        sources = None
        placeholder = st.empty()
        for piece in answer(question, history):
            if isinstance(piece, dict):  # 流末尾的来源信息
                sources = piece["__sources__"]
            else:
                full += piece
                placeholder.markdown(full + "▌")
        placeholder.markdown(full)

        if sources:
            with st.expander(f"📎 引用来源（{len(sources)} 条）"):
                for i, s in enumerate(sources, start=1):
                    page = f"，第 {s['page']} 页" if s["page"] else ""
                    st.markdown(f"**[{i}] {s['source']}{page}**")
                    st.text(s["text"][:300])

    st.session_state.messages.append({"role": "assistant", "content": full})
