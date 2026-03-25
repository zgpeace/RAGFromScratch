"""POST /upload — parse, chunk and index a Q&A text document."""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

from app.rag.loader import load_qa_text
from app.rag.embedder import add_documents

router = APIRouter()


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


@router.post("/upload", response_model=UploadResponse, summary="Upload a Q&A text document")
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a plain-text Q&A file, parse it into chunks and index them in FAISS."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    raw_bytes = await file.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    documents = load_qa_text(text, source=file.filename)
    if not documents:
        raise HTTPException(status_code=400, detail="No Q&A content found in the file.")

    add_documents(documents)
    return UploadResponse(
        filename=file.filename,
        chunks_indexed=len(documents),
        message=f"Successfully indexed {len(documents)} chunks from '{file.filename}'.",
    )
