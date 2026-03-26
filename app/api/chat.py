"""POST /chat — answer a question using the RAG pipeline."""

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.monitor.logger import log_request
from app.monitor.scorer import compute_scores
from app.rag.generator import generate_answer
from app.rag.retriever import is_index_ready, retrieve

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    scores: dict


@router.post("/chat", response_model=ChatResponse, summary="Ask a question via RAG")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    if not is_index_ready():
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload a document first.",
        )

    start = time.perf_counter()
    context_docs = retrieve(request.question, top_k=request.top_k)
    answer = generate_answer(request.question, context_docs)
    latency_ms = (time.perf_counter() - start) * 1000

    sources = [doc.metadata.get("source", "unknown") for doc in context_docs]
    unique_sources = list(dict.fromkeys(sources))  # preserve order, deduplicate

    scores = compute_scores(request.question, answer, context_docs, latency_ms)

    log_request(
        db=db,
        question=request.question,
        answer=answer,
        sources=unique_sources,
        **scores,
    )

    return ChatResponse(
        question=request.question,
        answer=answer,
        sources=unique_sources,
        scores=scores,
    )
