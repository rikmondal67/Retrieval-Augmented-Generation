from fastapi import APIRouter, Form, HTTPException
from pydantic import BaseModel, HttpUrl

from app.services.parsing import extract_markdown
from app.services.chunking import classify_structure, prepare_chunks
from app.services.storing import (
    setup_pinecone_index,
    setup_sparse_pinecone_index,
    ingest
)
from app.services.answering import ask


router = APIRouter()

pinecone_index = setup_pinecone_index()
sparse_index = setup_sparse_pinecone_index()


class AskRequest(BaseModel):
    question: str
    username: str


@router.post("/upload")
async def upload_pdf(
    username: str = Form(...),
    file_url: HttpUrl = Form(...)
):
    """
    Accepts a PDF URL, parses it into markdown via LlamaParse,
    structures and chunks it, and ingests it into Pinecone
    under the person's namespace.
    """

    file_url = str(file_url)

    try:
        # 1. Parse markdown from the remote PDF URL
        markdown_text = extract_markdown(file_url)

        # 2. Structure & Chunk
        structure = classify_structure(markdown_text)
        chunks = prepare_chunks(structure)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No extractable text or sections found in the PDF."
            )

        # 3. Ingest into Pinecone under the person's namespace
        ingested_chunks = ingest(
            chunks=chunks,
            username=username,
            pinecone_index=pinecone_index,
            sparse_index=sparse_index
        )

        return {
            "status": "success",
            "message": (
                f"Successfully ingested "
                f"{len(ingested_chunks)} chunks for {username}."
            ),
            "username": username,
            "file_url": file_url,
            "total_chunks": len(ingested_chunks)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/ask")
async def ask_question(payload: AskRequest):
    """
    Asks a question about an ingested document
    for a specific person.
    """

    try:
        answer = ask(
            user_question=payload.question,
            sparse_index=sparse_index,
            pinecone_index=pinecone_index,
            username=payload.username
        )

        return {
            "status": "success",
            "question": payload.question,
            "username": payload.username,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
