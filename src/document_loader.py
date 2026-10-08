"""文档加载与切分：支持 PDF / TXT / Markdown。"""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    text: str
    source: str       # 来源文件名
    page: int | None  # PDF 页码（从 1 开始），其他格式为 None


def _load_pdf(path: Path) -> list[tuple[str, int]]:
    """返回 [(页面文本, 页码)]，跳过提取不出文字的扫描页。"""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i))
    return pages


def load_file(path: Path) -> list[tuple[str, int | None]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in (".txt", ".md"):
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]
    return []


def split_into_chunks(path: Path) -> list[Chunk]:
    raw_pages = load_file(path)
    if not raw_pages:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for text, page in raw_pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(text=piece, source=path.name, page=page))
    return chunks
