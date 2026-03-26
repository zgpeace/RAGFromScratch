"""FAISS-backed vector store and retriever management."""

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.rag.embedder import get_embeddings

# Module-level singleton so all requests share the same index
_vector_store: Optional[FAISS] = None

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))


def get_vector_store() -> Optional[FAISS]:
    return _vector_store


def add_documents(docs: List[Document]) -> None:
    """Add documents to the FAISS vector store, creating it if necessary."""
    global _vector_store
    embeddings = get_embeddings()
    if _vector_store is None:
        _vector_store = FAISS.from_documents(docs, embeddings)
    else:
        _vector_store.add_documents(docs)


def retrieve(query: str, top_k: int = TOP_K) -> List[Document]:
    """Retrieve the top-k most relevant documents for the given query."""
    if _vector_store is None:
        return []
    retriever = _vector_store.as_retriever(search_kwargs={"k": top_k})
    return retriever.invoke(query)


def is_index_ready() -> bool:
    return _vector_store is not None
