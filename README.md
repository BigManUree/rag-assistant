# RAG 知识库问答助手

上传 PDF / TXT / Markdown 文档，基于文档内容进行问答，回答带编号引用与来源溯源。

## 技术架构

```
文档(PDF/TXT/MD)
   │  pypdf 解析
   ▼
文本切分（RecursiveCharacterTextSplitter，chunk=500，overlap=80）
   │  阿里云百炼 qwen3.7-text-embedding-flash 向量化（分批，单批 ≤20）
   ▼
Chroma 向量库（本地持久化，余弦相似度）
   │  用户提问 -> 向量检索 Top-4
   ▼
Prompt 组装（检索片段 + 最近 3 轮对话历史）
   │  qwen3.8-flash，temperature=0.3，流式输出
   ▼
带 [1][2] 引用的回答 + 来源面板
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY 与 LLM_BASE_URL（阿里云百炼 MaaS，OpenAI 兼容模式）

# 3. 启动
streamlit run app.py
```

也可以把文档放进 `data/` 后用命令行批量导入：

```bash
python ingest.py
```

## 目录结构

| 路径 | 说明 |
|---|---|
| `app.py` | Streamlit 界面 |
| `ingest.py` | 命令行批量导入 |
| `src/config.py` | 配置与模型客户端 |
| `src/document_loader.py` | 文档解析与切分 |
| `src/vector_store.py` | embedding 与 Chroma 向量库 |
| `src/rag.py` | 检索 + 生成核心逻辑 |
| `data/` | 原始文档 |
