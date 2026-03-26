"""Persist request/response logs and quality scores to SQLite."""
from __future__ import annotations

import json
from typing import List

from sqlalchemy.orm import Session

from app.database.models import RequestLog


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
    """Insert a new log entry and return the persisted ORM object."""
    entry = RequestLog(
        question=question,
        answer=answer,
        sources=json.dumps(sources, ensure_ascii=False),
        relevance_score=relevance_score,
        faithfulness_score=faithfulness_score,
        completeness_score=completeness_score,
        latency_ms=latency_ms,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
