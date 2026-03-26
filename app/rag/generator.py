"""Answer generation using gpt-4o-mini via OpenAI-compatible API."""

import os
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def _build_chain() -> object:
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant. Answer the user's question using ONLY "
                    "the context provided below. If the context does not contain enough "
                    "information to answer, say so clearly.\n\n"
                    "Context:\n{context}"
                ),
            ),
            ("human", "{question}"),
        ]
    )

    return prompt | llm | StrOutputParser()


def generate_answer(question: str, context_docs: List[Document]) -> str:
    """Generate an answer for *question* grounded in *context_docs*."""
    context = "\n\n".join(doc.page_content for doc in context_docs)
    chain = _build_chain()
    return chain.invoke({"question": question, "context": context})
