"""Embedding and FAISS vector-store management."""
from __future__ import annotations

import os
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Module-level singleton so the index persists across requests within one process
_vector_store: Optional[FAISS] = None


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )


def add_documents(documents: List[Document]) -> None:
    """Embed *documents* and upsert them into the in-memory FAISS store."""
    global _vector_store
    embeddings = _get_embeddings()
    if _vector_store is None:
        _vector_store = FAISS.from_documents(documents, embeddings)
    else:
        _vector_store.add_documents(documents)


def get_vector_store() -> Optional[FAISS]:
    return _vector_store


def is_ready() -> bool:
    return _vector_store is not None
