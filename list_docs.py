"""查看知识库里有哪些文档 / 文本块。

用法：
    python list_docs.py                 # 按文档汇总：块数、页码范围
    python list_docs.py --chunks        # 全部文本块逐条预览（前 60 字）
    python list_docs.py --doc 护理员工工作手册.pdf   # 只看某个文档的文本块
    python list_docs.py --json          # 输出 JSON，便于脚本处理
"""
import argparse
import json
from collections import Counter, defaultdict

from src.vector_store import get_collection


def _load():
    collection = get_collection()
    data = collection.get(include=["metadatas", "documents"])
    metas = data.get("metadatas") or []
    docs = data.get("documents") or []
    rows = []
    for text, meta in zip(docs, metas):
        meta = meta or {}
        rows.append({
            "source": meta.get("source", "?"),
            "page": meta.get("page") or 0,
            "text": text or "",
        })
    return collection, rows


def main():
    parser = argparse.ArgumentParser(description="查看 Chroma 知识库内容")
    parser.add_argument("--doc", help="按文件名过滤")
    parser.add_argument("--chunks", action="store_true", help="逐块预览")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--max-chars", type=int, default=60, help="每块预览字数")
    args = parser.parse_args()

    collection, rows = _load()
    if args.doc:
        rows = [r for r in rows if r["source"] == args.doc]
        if not rows:
            raise SystemExit(f"知识库里没有名为 {args.doc!r} 的文档")

    if args.json:
        print(json.dumps({"count": len(rows), "chunks": rows},
                         ensure_ascii=False, indent=2))
        return

    print(f"集合：{collection.name}    文本块总数：{collection.count()}")
    print(f"本次列出：{len(rows)} 块\n")

    if not args.chunks:
        by_source = defaultdict(list)
        for r in rows:
            by_source[r["source"]].append(r["page"])
        print(f"{'文档':<40}{'块数':>6}   页码")
        print("-" * 78)
        for source, pages in sorted(by_source.items(),
                                    key=lambda kv: -len(kv[1])):
            valid = sorted({p for p in pages if p})
            page_info = (f"{valid[0]}-{valid[-1]}（{len(valid)} 页）"
                         if valid else "无页码")
            print(f"{source:<40}{len(pages):>6}   {page_info}")
        print("\n加 --chunks 查看具体内容，加 --doc <文件名> 只看一份文档。")
        return

    for i, r in enumerate(rows, start=1):
        page = f" p{r['page']}" if r["page"] else ""
        preview = r["text"][:args.max_chars].replace("\n", " ")
        print(f"[{i:03d}] {r['source']}{page} | {preview}")


if __name__ == "__main__":
    main()
