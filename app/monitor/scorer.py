"""Multi-dimensional quality scoring for RAG responses."""

import os
from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        temperature=0,
    )


def _ask_score(prompt: str) -> float:
    """Ask the LLM to return a single float score between 0 and 1."""
    response = _llm().invoke(prompt)
    text = response.content.strip()
    try:
        score = float(text.split()[0])
        return max(0.0, min(1.0, score))
    except (ValueError, IndexError):
        return 0.5


def score_relevance(question: str, answer: str) -> float:
    """Score how relevant the answer is to the question (0–1)."""
    prompt = (
        f"Rate how relevant the following answer is to the question on a scale of 0.0 to 1.0. "
        f"Reply with ONLY the numeric score.\n\n"
        f"Question: {question}\nAnswer: {answer}"
    )
    return _ask_score(prompt)


def score_faithfulness(answer: str, context_docs: List[Document]) -> float:
    """Score how faithful the answer is to the retrieved context (0–1)."""
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = (
        f"Rate how faithful the following answer is to the context provided, on a scale of 0.0 "
        f"to 1.0. A score of 1.0 means the answer is entirely supported by the context. "
        f"Reply with ONLY the numeric score.\n\n"
        f"Context:\n{context}\n\nAnswer: {answer}"
    )
    return _ask_score(prompt)


def score_completeness(question: str, answer: str) -> float:
    """Score how completely the answer addresses the question (0–1)."""
    prompt = (
        f"Rate how completely the following answer addresses the question on a scale of 0.0 to "
        f"1.0. Reply with ONLY the numeric score.\n\n"
        f"Question: {question}\nAnswer: {answer}"
    )
    return _ask_score(prompt)


def compute_scores(
    question: str,
    answer: str,
    context_docs: List[Document],
    latency_ms: float,
) -> dict:
    """Compute all quality dimensions and return as a dict."""
    return {
        "relevance_score": score_relevance(question, answer),
        "faithfulness_score": score_faithfulness(answer, context_docs),
        "completeness_score": score_completeness(question, answer),
        "latency_ms": latency_ms,
    }
