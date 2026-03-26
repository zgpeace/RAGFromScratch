"""FAISS-based retriever that returns the top-k most similar documents."""
from __future__ import annotations

from typing import List

from langchain.schema import Document

from app.rag.embedder import get_vector_store

TOP_K = 3


def retrieve(query: str, k: int = TOP_K) -> List[Document]:
    """Return the *k* most relevant documents for *query*.

    Returns an empty list if the vector store has not been populated yet.
    """
    store = get_vector_store()
    if store is None:
        return []
    return store.similarity_search(query, k=k)
