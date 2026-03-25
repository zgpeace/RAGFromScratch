"""Multi-dimensional quality scoring for RAG responses.

All scores are in the range [0.0, 1.0].  The current implementation uses
lightweight heuristics so that the application works without an additional LLM
call for scoring.  Replace the individual scorer functions with LLM-based or
embedding-based approaches as needed.
"""
from __future__ import annotations

import re
from typing import List

from langchain.schema import Document


def score_relevance(question: str, answer: str) -> float:
    """Estimate how relevant the answer is to the question.

    Heuristic: count how many non-stopword tokens from the question appear in
    the answer.
    """
    stopwords = {"what", "is", "are", "the", "a", "an", "of", "in", "to", "how", "does", "do"}
    q_tokens = {t.lower() for t in re.findall(r"\w+", question) if t.lower() not in stopwords}
    if not q_tokens:
        return 0.5
    a_text = answer.lower()
    matched = sum(1 for t in q_tokens if t in a_text)
    return round(min(matched / len(q_tokens), 1.0), 4)


def score_faithfulness(answer: str, documents: List[Document]) -> float:
    """Estimate how faithfully the answer sticks to the retrieved documents.

    Heuristic: count what fraction of answer sentences contain at least one
    token that also appears in the retrieved context.
    """
    if not documents:
        return 0.0
    context = " ".join(doc.page_content for doc in documents).lower()
    context_tokens = set(re.findall(r"\w+", context))
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.5
    faithful_count = 0
    for sentence in sentences:
        s_tokens = set(re.findall(r"\w+", sentence.lower()))
        if s_tokens & context_tokens:
            faithful_count += 1
    return round(faithful_count / len(sentences), 4)


def score_completeness(answer: str) -> float:
    """Estimate answer completeness based on length and structure.

    Heuristic: answers longer than 50 words score 1.0; shorter answers score
    proportionally less.
    """
    word_count = len(re.findall(r"\w+", answer))
    return round(min(word_count / 50.0, 1.0), 4)


def compute_scores(
    question: str,
    answer: str,
    documents: List[Document],
    latency_ms: float,
) -> dict:
    """Return a dictionary of all quality scores."""
    return {
        "relevance_score": score_relevance(question, answer),
        "faithfulness_score": score_faithfulness(answer, documents),
        "completeness_score": score_completeness(answer),
        "latency_ms": round(latency_ms, 2),
    }
