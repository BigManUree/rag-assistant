"""命令行入口：把 data/ 目录下的文档灌入向量库。

用法： python ingest.py
"""
from pathlib import Path

from src.config import DATA_DIR, ZHIPU_API_KEY
from src.document_loader import split_into_chunks
from src.vector_store import add_chunks

SUPPORTED = {".pdf", ".txt", ".md"}


def main():
    if not ZHIPU_API_KEY or ZHIPU_API_KEY == "your_key_here":
        raise SystemExit("请先在 .env 中配置 LLM_API_KEY（参考 .env.example）")

    files = [p for p in DATA_DIR.iterdir() if p.suffix.lower() in SUPPORTED]
    if not files:
        raise SystemExit(f"data/ 目录下没有可导入的文档（支持 {SUPPORTED}）")

    total = 0
    for path in files:
        chunks = split_into_chunks(path)
        count = add_chunks(chunks)
        total += count
        print(f"  {path.name}: {count} 个文本块")
    print(f"完成，共写入 {total} 个文本块。")


if __name__ == "__main__":
    main()
