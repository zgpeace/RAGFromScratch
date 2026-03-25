"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # Load .env before importing modules that read env vars

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import upload, chat, monitor
from app.database.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="RAGFromScratch API",
    description=(
        "A complete Retrieval-Augmented Generation (RAG) application with "
        "document upload, Q&A chat, and request quality monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit (running on a different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload.router, tags=["Documents"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(monitor.router, tags=["Monitor"])


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "RAGFromScratch API is running."}
