"""集中管理配置：从 .env 读取 API Key 和模型名称。"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 项目根目录（src 的上一级）
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"

load_dotenv(ROOT_DIR / ".env")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "glm-4-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")

# 智谱提供 OpenAI 兼容接口，直接用 openai SDK 即可
client = OpenAI(
    # SDK 不接受空 key，未配置时用占位值，由 app.py/ingest.py 在调用前检查
    api_key=ZHIPU_API_KEY or "missing-credentials",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# 切分参数：面试常考点——chunk 太大噪声多、太小语义不完整；overlap 避免句子被切断
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4
