"""LLM-based answer generation using retrieved context documents."""
from __future__ import annotations

import os
from typing import List

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "context provided below. If the context does not contain enough information "
    "to answer the question, say so honestly.\n\n"
    "Context:\n{context}"
)

USER_PROMPT = "{question}"


def _build_context(documents: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


def generate_answer(question: str, documents: List[Document]) -> str:
    """Generate an answer to *question* grounded in *documents*."""
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    context = _build_context(documents)
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": question})
    return result.content
