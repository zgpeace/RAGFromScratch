# RAGFromScratch

A complete **Retrieval-Augmented Generation (RAG)** application built with FastAPI, LangChain, FAISS, and Streamlit.

---

## 功能概述 / Overview

| Feature | Description |
|---------|-------------|
| 📤 Document Upload | Parse and index Q&A `.txt` files into a FAISS vector store |
| 💬 RAG Chat | Retrieve relevant context and generate answers with `gpt-4o-mini` |
| 📚 Source Inspection | View all indexed document chunks |
| 📊 Quality Monitoring | Multi-dimensional scoring (relevance, faithfulness, completeness, latency) stored in SQLite |
| 🖥️ Streamlit UI | Interactive web interface for all features |
| 📖 Auto Docs | FastAPI `/docs` page with full API documentation |

---

## 项目结构 / Project Structure

```
RAGFromScratch/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── upload.py        # POST /upload
│   │   ├── chat.py          # POST /chat
│   │   └── monitor.py       # GET /history, GET /sources
│   ├── rag/
│   │   ├── loader.py        # Q&A text parser
│   │   ├── embedder.py      # OpenAI embeddings wrapper
│   │   ├── retriever.py     # FAISS vector store & retrieval
│   │   └── generator.py     # gpt-4o-mini answer generation
│   ├── monitor/
│   │   ├── logger.py        # SQLite persistence
│   │   └── scorer.py        # Multi-dimensional quality scoring
│   └── database/
│       ├── models.py        # SQLAlchemy models
│       └── db.py            # Database engine & session
├── ui/
│   └── streamlit_app.py     # Streamlit web interface
├── data/
│   └── sample_qa.txt        # Sample Q&A document
├── requirements.txt
├── .env.example
└── README.md
```

---

## 快速开始 / Quick Start

### 1. Clone the repository / 克隆仓库

```bash
git clone https://github.com/zgpeace/RAGFromScratch.git
cd RAGFromScratch
```

### 2. Create a virtual environment / 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies / 安装依赖

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables / 配置环境变量

```bash
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY
```

### 5. Start the FastAPI backend / 启动后端

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 6. Start the Streamlit UI / 启动前端界面

Open a second terminal:

```bash
streamlit run ui/streamlit_app.py
```

The UI will open at [http://localhost:8501](http://localhost:8501).

---

## API 接口 / API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload a Q&A `.txt` file for indexing |
| `POST` | `/chat` | Send a question and receive a RAG-generated answer |
| `GET` | `/history` | Retrieve historical Q&A records with quality scores |
| `GET` | `/sources` | List all indexed document chunks |
| `GET` | `/` | Health check |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |

### Example: Chat Request / 问答示例

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 4}'
```

**Response:**

```json
{
  "question": "What is RAG?",
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "sources": ["sample_qa.txt"],
  "scores": {
    "relevance_score": 0.95,
    "faithfulness_score": 0.92,
    "completeness_score": 0.88,
    "latency_ms": 1234.5
  }
}
```

### Example: Upload Document / 上传文档示例

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/sample_qa.txt"
```

---

## Q&A 文档格式 / Q&A Document Format

The uploaded `.txt` file must use the following format:

```
Q: What is RAG?
A: RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval of relevant documents with language model generation.

Q: What is LangChain?
A: LangChain is an open-source framework for building applications powered by large language models.
```

A sample file is provided at `data/sample_qa.txt`.

---

## 质量评分维度 / Quality Score Dimensions

| Dimension | Description |
|-----------|-------------|
| **Relevance** | How relevant the answer is to the question (0–1) |
| **Faithfulness** | How well the answer is grounded in retrieved context (0–1) |
| **Completeness** | How completely the answer addresses the question (0–1) |
| **Latency (ms)** | End-to-end response time in milliseconds |

Scores are computed by the LLM and stored in SQLite for every request.

---

## 技术栈 / Tech Stack

- **Backend Framework**: FastAPI
- **RAG Framework**: LangChain
- **Vector Database**: FAISS
- **Embedding Model**: `text-embedding-ada-002` (OpenAI)
- **LLM**: `gpt-4o-mini` (OpenAI)
- **Monitoring Storage**: SQLite + SQLAlchemy
- **Frontend**: Streamlit
- **Dependency Management**: `requirements.txt`

---

## License

MIT