"""GET /history and GET /sources monitoring endpoints."""

import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import RequestLog, SourceDocument

router = APIRouter()


@router.get("/history", summary="Get historical request/response records with scores")
def get_history(limit: int = 50, db: Session = Depends(get_db)):
    records = (
        db.query(RequestLog)
        .order_by(RequestLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "question": r.question,
            "answer": r.answer,
            "sources": json.loads(r.sources) if r.sources else [],
            "relevance_score": r.relevance_score,
            "faithfulness_score": r.faithfulness_score,
            "completeness_score": r.completeness_score,
            "latency_ms": r.latency_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]


@router.get("/sources", summary="Get all indexed source document chunks")
def get_sources(limit: int = 200, db: Session = Depends(get_db)):
    docs = (
        db.query(SourceDocument)
        .order_by(SourceDocument.created_at.desc(), SourceDocument.chunk_index)
        .limit(limit)
        .all()
    )
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "chunk_index": d.chunk_index,
            "content": d.content,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]
