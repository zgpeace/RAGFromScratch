# RAGFromScratch

A complete **Retrieval-Augmented Generation (RAG)** application built with FastAPI, LangChain, FAISS, and Streamlit.

---

## Features / 功能特性

| Feature | Description |
|---------|-------------|
| 📤 **Document Upload** | Upload Q&A text files and index them into FAISS |
| 💬 **RAG Chat** | Ask questions and get grounded answers from `gpt-4o-mini` |
| 📄 **Source Retrieval** | See which documents were used to generate each answer |
| 📊 **Quality Monitoring** | Track relevance, faithfulness, completeness and latency per request |
| 🗄️ **Persistent Logs** | All requests and scores stored in SQLite |
| 🖥️ **Web UI** | Streamlit interface for upload, chat, sources and monitoring |

---

## Project Structure / 项目结构

```
RAGFromScratch/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── upload.py        # POST /upload
│   │   ├── chat.py          # POST /chat
│   │   └── monitor.py       # GET /history, GET /sources
│   ├── rag/
│   │   ├── loader.py        # Parse Q&A text files
│   │   ├── embedder.py      # FAISS vector store management
│   │   ├── retriever.py     # Similarity search
│   │   └── generator.py     # LLM answer generation
│   ├── monitor/
│   │   ├── logger.py        # Persist logs to SQLite
│   │   └── scorer.py        # Multi-dimensional quality scoring
│   └── database/
│       ├── models.py        # SQLAlchemy ORM models
│       └── db.py            # Database connection & session
├── ui/
│   └── streamlit_app.py     # Streamlit web interface
├── data/
│   └── sample_qa.txt        # Sample Q&A data
├── requirements.txt
├── .env.example
└── README.md
```

---

## Prerequisites / 前置要求

- Python 3.10+
- An OpenAI API key (or compatible endpoint)

---

## Installation / 安装

```bash
# Clone the repository
git clone https://github.com/zgpeace/RAGFromScratch.git
cd RAGFromScratch

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration / 配置

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-ada-002
```

---

## Running the Application / 运行应用

### 1. Start the FastAPI backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs

### 2. Start the Streamlit UI (in a separate terminal)

```bash
streamlit run ui/streamlit_app.py
```

The UI will be available at http://localhost:8501

---

## API Endpoints / API 接口

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload a Q&A text document |
| `POST` | `/chat` | Ask a question via RAG |
| `GET` | `/history` | Get request history with quality scores |
| `GET` | `/sources` | List indexed document sources |

### POST /upload

Upload a plain-text Q&A file:

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/sample_qa.txt"
```

### POST /chat

Ask a question:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 3}'
```

Response:

```json
{
  "question": "What is RAG?",
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "sources": ["sample_qa.txt"],
  "relevance_score": 0.9,
  "faithfulness_score": 0.85,
  "completeness_score": 0.8,
  "latency_ms": 1234.5
}
```

### GET /history

```bash
curl "http://localhost:8000/history?limit=10"
```

### GET /sources

```bash
curl http://localhost:8000/sources
```

---

## Q&A File Format / Q&A 文件格式

Each Q&A pair should be separated by a blank line:

```
Q: What is RAG?
A: RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval of relevant documents with language model generation.

Q: What are the benefits of RAG?
A: RAG improves accuracy by grounding responses in retrieved documents, reducing hallucinations and improving factual correctness.
```

---

## Quality Metrics / 质量评分指标

| Metric | Description |
|--------|-------------|
| **Relevance** | How relevant the answer is to the question (token overlap heuristic) |
| **Faithfulness** | How faithfully the answer sticks to the retrieved context |
| **Completeness** | Length-based estimate of answer completeness |
| **Latency (ms)** | End-to-end response time in milliseconds |

All scores are in the range **[0.0, 1.0]**.

---

## Tech Stack / 技术栈

| Component | Technology |
|-----------|-----------|
| Backend framework | FastAPI |
| RAG framework | LangChain |
| Vector database | FAISS |
| Embedding model | `text-embedding-ada-002` |
| LLM | `gpt-4o-mini` via OpenAI API |
| Monitoring storage | SQLite + SQLAlchemy |
| Web UI | Streamlit |

---

## License

MIT