"""Persist request/response records to SQLite."""

import json
from typing import List

from sqlalchemy.orm import Session

from app.database.models import RequestLog, SourceDocument


def log_request(
    db: Session,
    question: str,
    answer: str,
    sources: List[str],
    relevance_score: float,
    faithfulness_score: float,
    completeness_score: float,
    latency_ms: float,
) -> RequestLog:
    """Write one request/response record to the database."""
    record = RequestLog(
        question=question,
        answer=answer,
        sources=json.dumps(sources),
        relevance_score=relevance_score,
        faithfulness_score=faithfulness_score,
        completeness_score=completeness_score,
        latency_ms=latency_ms,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def save_source_documents(
    db: Session,
    filename: str,
    chunks: List[str],
) -> None:
    """Persist the text chunks of an uploaded document to the database."""
    for idx, content in enumerate(chunks):
        doc = SourceDocument(
            filename=filename,
            chunk_index=idx,
            content=content,
        )
        db.add(doc)
    db.commit()
