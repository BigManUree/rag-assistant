"""向量库封装：调用智谱 embedding 接口，存入本地 Chroma。"""
import hashlib

import chromadb

from src.config import CHROMA_DIR, EMBEDDING_MODEL, client
from src.document_loader import Chunk


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


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
