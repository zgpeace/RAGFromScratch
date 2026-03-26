"""Parse Q&A text files and split them into LangChain Document objects."""
from __future__ import annotations

import re
from typing import List

from langchain.schema import Document


def load_qa_text(text: str, source: str = "uploaded") -> List[Document]:
    """Convert raw Q&A text content into a list of LangChain Documents.

    Each Q/A pair becomes one Document whose page_content is the combined
    question and answer, and whose metadata carries the source filename.
    """
    documents: List[Document] = []
    # Split on blank lines between Q/A pairs
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Accept blocks that start with "Q:" even if they span multiple lines
        if block.upper().startswith("Q:"):
            documents.append(
                Document(page_content=block, metadata={"source": source})
            )
        else:
            # Include non-Q&A blocks as plain documents
            documents.append(
                Document(page_content=block, metadata={"source": source})
            )
    return documents
