"""FastAPI application entry point."""

from dotenv import load_dotenv

load_dotenv()  # load .env before any module reads env vars

from fastapi import FastAPI  # noqa: E402

from app.api import chat, monitor, upload  # noqa: E402
from app.database.db import init_db  # noqa: E402

app = FastAPI(
    title="RAGFromScratch API",
    description=(
        "A Retrieval-Augmented Generation (RAG) application with Q&A document ingestion, "
        "FAISS vector retrieval, gpt-4o-mini generation, and multi-dimensional quality monitoring."
    ),
    version="1.0.0",
)

# Initialise SQLite tables on startup
init_db()

# Register routers
app.include_router(upload.router, tags=["Upload"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(monitor.router, tags=["Monitor"])


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "RAGFromScratch API is running. Visit /docs for the API documentation."}
