"""GET /history and GET /sources monitoring endpoints."""
from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import RequestLog
from app.rag.embedder import get_vector_store

router = APIRouter()


class LogEntry(BaseModel):
    id: int
    question: str
    answer: str
    sources: List[str]
    relevance_score: Optional[float]
    faithfulness_score: Optional[float]
    completeness_score: Optional[float]
    latency_ms: Optional[float]
    created_at: str

    class Config:
        from_attributes = True


@router.get("/history", response_model=List[LogEntry], summary="Get request history with scores")
def get_history(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> List[LogEntry]:
    """Return the most recent *limit* request/response log entries."""
    rows = (
        db.query(RequestLog)
        .order_by(RequestLog.created_at.desc())
        .limit(limit)
        .all()
    )
    result = []
    for row in rows:
        result.append(
            LogEntry(
                id=row.id,
                question=row.question,
                answer=row.answer,
                sources=json.loads(row.sources) if row.sources else [],
                relevance_score=row.relevance_score,
                faithfulness_score=row.faithfulness_score,
                completeness_score=row.completeness_score,
                latency_ms=row.latency_ms,
                created_at=str(row.created_at),
            )
        )
    return result


class SourceInfo(BaseModel):
    source: str
    num_chunks: int


@router.get("/sources", response_model=List[SourceInfo], summary="List indexed document sources")
def get_sources() -> List[SourceInfo]:
    """Return unique source filenames and the number of chunks indexed from each."""
    store = get_vector_store()
    if store is None:
        return []

    source_counts: dict[str, int] = {}
    # FAISS docstore stores Document objects keyed by integer index
    for doc_id in store.index_to_docstore_id.values():
        doc = store.docstore.search(doc_id)
        if doc and hasattr(doc, "metadata"):
            src = doc.metadata.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

    return [SourceInfo(source=src, num_chunks=cnt) for src, cnt in source_counts.items()]
