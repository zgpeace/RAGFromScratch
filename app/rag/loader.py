"""Parse Q&A text files into LangChain Document objects."""

import re
from typing import List

from langchain_core.documents import Document


def parse_qa_text(text: str, source: str = "uploaded") -> List[Document]:
    """Split a Q&A-formatted text into individual Document chunks.

    Each Q/A pair becomes one Document whose page_content is
    ``"Q: ...\\nA: ..."`` and whose metadata records the source filename
    and a sequential chunk index.
    """
    # Split on lines that start a new question (^Q:)
    pattern = re.compile(r"(?=^Q:)", re.MULTILINE)
    raw_chunks = pattern.split(text)

    docs: List[Document] = []
    chunk_index = 0
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        docs.append(
            Document(
                page_content=chunk,
                metadata={"source": source, "chunk_index": chunk_index},
            )
        )
        chunk_index += 1

    return docs
