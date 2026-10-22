"""向量库封装：调用 OpenAI 兼容 embedding 接口（自动分批），存入本地 Chroma。"""
import hashlib
import os

import chromadb

from src.config import CHROMA_DIR, EMBEDDING_MODEL, client
from src.document_loader import Chunk

# 百炼/通义 embedding 单次 input 不能超过 20 条，留出余量默认 10，可用环境变量覆盖
BATCH_SIZE = max(1, int(os.getenv("EMBED_BATCH_SIZE", "10")))


def _is_batch_too_large(err: Exception) -> bool:
    """识别“批次过大 / 长度超限”类 400，用于自动二分降批。"""
    msg = str(err).lower()
    if "400" not in msg and "invalidparameter" not in msg and "invalid_parameter" not in msg:
        return False
    return any(k in msg for k in ("batch", "too large", "larger than", "too long", "length"))


def _embed_batch(batch: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    data = list(resp.data)
    # 兼容服务端可能乱序返回：按 index 排序，保证向量与文本一一对应
    if data and getattr(data[0], "index", None) is not None:
        data.sort(key=lambda d: d.index)
    if len(data) != len(batch):
        raise RuntimeError(f"embedding 返回数量不匹配：期望 {len(batch)}，实际 {len(data)}")
    return [item.embedding for item in data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """分批向量化。

    1. 按 BATCH_SIZE 切片，逐批请求，结果按原顺序拼接；
    2. 若某批仍被判定为过大，自动对半降批重试，直到单条；
    3. 单条仍失败则原样抛出，不做静默截断。
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    size = BATCH_SIZE
    i = 0
    while i < len(texts):
        batch = texts[i:i + size]
        try:
            vectors = _embed_batch(batch)
        except Exception as exc:
            if len(batch) > 1 and _is_batch_too_large(exc):
                size = max(1, len(batch) // 2)
                continue
            raise
        all_embeddings.extend(vectors)
        i += len(batch)
    return all_embeddings


def get_collection():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma_client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"},  # 余弦相似度
    )


def _chunk_id(chunk: Chunk, index: int) -> str:
    raw = f"{chunk.source}:{chunk.page}:{index}:{chunk.text[:64]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def add_chunks(chunks: list[Chunk]) -> int:
    """将切分后的文档写入向量库，返回新增/更新的块数。"""
    if not chunks:
        return 0
    collection = get_collection()
    embeddings = embed_texts([c.text for c in chunks])
    collection.upsert(
        ids=[_chunk_id(c, i) for i, c in enumerate(chunks)],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c.source, "page": c.page or 0} for c in chunks],
    )
    return len(chunks)


def query_similar(question: str, top_k: int) -> list[dict]:
    """检索与问题最相似的 top_k 个文本块。"""
    collection = get_collection()
    result = collection.query(
        query_embeddings=embed_texts([question]),
        n_results=top_k,
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    return [{"text": t, "source": m.get("source", "?"),
             "page": m.get("page") or None} for t, m in zip(docs, metas)]
