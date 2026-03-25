"""POST /chat — ask a question and get a RAG-generated answer."""
from __future__ import annotations

import time
import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.monitor.logger import log_request
from app.monitor.scorer import compute_scores
from app.rag.embedder import is_ready
from app.rag.generator import generate_answer
from app.rag.retriever import retrieve

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    top_k: int = 3


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    relevance_score: float
    faithfulness_score: float
    completeness_score: float
    latency_ms: float


@router.post("/chat", response_model=ChatResponse, summary="Ask a question via RAG")
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Retrieve relevant documents and generate a grounded answer."""
    if not is_ready():
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload a document first.",
        )

    start = time.time()
    docs = retrieve(request.question, k=request.top_k)
    answer = generate_answer(request.question, docs)
    latency_ms = (time.time() - start) * 1000

    sources = list({doc.metadata.get("source", "unknown") for doc in docs})
    scores = compute_scores(request.question, answer, docs, latency_ms)

    log_request(
        db=db,
        question=request.question,
        answer=answer,
        sources=sources,
        **scores,
    )

    return ChatResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        **scores,
    )
