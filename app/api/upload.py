"""POST /upload — ingest a Q&A text document into the vector store."""

import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.monitor.logger import save_source_documents
from app.rag.loader import parse_qa_text
from app.rag.retriever import add_documents

router = APIRouter()


@router.post("/upload", summary="Upload a Q&A text document")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    raw_bytes = await file.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")

    docs = parse_qa_text(text, source=file.filename)
    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No Q&A pairs found. Ensure the file uses 'Q: ...' / 'A: ...' format.",
        )

    add_documents(docs)

    chunk_contents = [doc.page_content for doc in docs]
    save_source_documents(db, filename=file.filename, chunks=chunk_contents)

    return {
        "filename": file.filename,
        "chunks_indexed": len(docs),
        "message": f"Successfully indexed {len(docs)} Q&A chunks.",
    }
