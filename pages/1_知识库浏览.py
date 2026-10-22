"""知识库浏览页：查看库里有哪些文档、每份文档的文本块，支持关键词检索与删除。

Streamlit 多页应用：本文件放在 pages/ 下，启动后左侧栏会出现「知识库浏览」入口。
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import DATA_DIR
from src.vector_store import get_collection


@st.cache_data(ttl=3, show_spinner=False)
def load_rows():
    """读出全部块：[(source, page, text)]，按来源、页码排序。"""
    collection = get_collection()
    data = collection.get(include=["metadatas", "documents"])
    rows = []
    for text, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
        meta = meta or {}
        rows.append({
            "source": meta.get("source", "?"),
            "page": meta.get("page") or 0,
            "text": text or "",
        })
    rows.sort(key=lambda r: (r["source"], r["page"]))
    return rows


def page_range(pages: list[int]) -> str:
    valid = sorted({p for p in pages if p})
    if not valid:
        return "无页码"
    return f"{valid[0]}-{valid[-1]}（{len(valid)} 页）"


rows = load_rows()
by_source = defaultdict(list)
for row in rows:
    by_source[row["source"]].append(row)

st.title("📂 知识库内容浏览")
st.caption("查看当前 Chroma 集合 `knowledge_base` 里的文档、文本块与检索预览；"
           "本页只读，删除操作在左侧「维护」里。")

if not rows:
    st.info("知识库还是空的。回到「RAG 知识库问答」页面上传文档，或运行 `python ingest.py`。")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("文档数", len(by_source))
c2.metric("文本块数", len(rows))
avg = sum(len(r["text"]) for r in rows) // max(1, len(rows))
c3.metric("平均块长度", f"{avg} 字")

# ---------- 侧边栏：过滤与维护 ----------
with st.sidebar:
    st.header("筛选")
    keyword = st.text_input("关键词（在块内容中匹配）", placeholder="如：排班 / 高血压")
    only_doc = st.selectbox("只看某个文档", ["全部文档"] + sorted(by_source))

    st.divider()
    with st.expander("🗑 删除文档"):
        target = st.selectbox("选择要删除的文档", sorted(by_source), key="del_doc")
        also_file = st.checkbox(f"同时删除 data/ 下的原文件", value=False)
        confirm = st.checkbox("我确认要从知识库移除该文档", key="del_confirm")
        if st.button("执行删除", type="secondary", disabled=not confirm):
            removed = len(by_source[target])
            try:
                get_collection().delete(where={"source": target})
                if also_file:
                    path = DATA_DIR / target
                    if path.exists():
                        path.unlink()
                st.cache_data.clear()
                st.success(f"已移除 {target} 的 {removed} 个文本块"
                           + ("，原文件已删除" if also_file else ""))
                st.rerun()
            except Exception as exc:
                st.error(f"删除失败：{type(exc).__name__}: {exc}")

# ---------- 汇总表 ----------
st.subheader("文档清单")
summary = [
    {
        "文档": source,
        "文本块": len(items),
        "页码范围": page_range([i["page"] for i in items]),
        "总字数": sum(len(i["text"]) for i in items),
    }
    for source, items in sorted(by_source.items(), key=lambda kv: -len(kv[1]))
]
st.dataframe(summary, width="stretch", hide_index=True)

# ---------- 关键词检索结果 ----------
filtered = rows
if keyword.strip():
    kw = keyword.strip().lower()
    filtered = [r for r in rows if kw in r["text"].lower()]
    st.subheader(f"关键词「{keyword}」命中 {len(filtered)} 块")
    if not filtered:
        st.warning("没有块包含这个关键词。注意这是字面匹配，不是语义检索。")
elif only_doc != "全部文档":
    filtered = [r for r in filtered if r["source"] == only_doc]

if keyword.strip() and filtered:
    for i, r in enumerate(filtered[:100], start=1):
        page = f"第 {r['page']} 页" if r["page"] else "无页码"
        with st.expander(f"[{i}] {r['source']} · {page} · {len(r['text'])} 字"):
            st.text_area("内容", r["text"], height=180, disabled=True,
                         key=f"kw_{i}", label_visibility="collapsed")
    if len(filtered) > 100:
        st.caption(f"仅显示前 100 条，共 {len(filtered)} 条命中。")

# ---------- 单文档逐块浏览 ----------
st.divider()
st.subheader("逐块浏览")
doc_options = sorted(by_source) if only_doc == "全部文档" else [only_doc]
doc = st.selectbox("选择文档", doc_options, key="view_doc")
doc_rows = by_source[doc]

top = st.columns([1, 1, 2])
page_size = top[0].selectbox("每页块数", [10, 20, 50, 100], key="page_size")
total_pages = max(1, (len(doc_rows) + page_size - 1) // page_size)
page_no = top[1].number_input("页码", min_value=1, max_value=total_pages, value=1, step=1)
top[2].download_button(
    "⬇ 导出该文档全部块（txt）",
    data="\n\n".join(f"[{i}] 第{r['page'] or '-'}页\n{r['text']}"
                     for i, r in enumerate(doc_rows, start=1)),
    file_name=f"{Path(doc).stem}_chunks.txt",
    mime="text/plain",
    width="stretch",
)

start = (int(page_no) - 1) * page_size
for idx, r in enumerate(doc_rows[start:start + page_size], start=start + 1):
    page = f"第 {r['page']} 页" if r["page"] else "无页码"
    with st.expander(f"#{idx:03d} · {page} · {len(r['text'])} 字"):
        st.text_area("内容", r["text"], height=200, disabled=True,
                     key=f"doc_{doc}_{idx}", label_visibility="collapsed")

st.caption(f"{doc}：共 {len(doc_rows)} 块，第 {page_no}/{total_pages} 页")
