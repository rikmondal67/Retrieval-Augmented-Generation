import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from app.services.parsing import extract_markdown
from app.services.chunking import classify_structure, prepare_chunks
from app.services.storing import setup_pinecone_index, setup_sparse_pinecone_index, ingest
from app.services.answering import ask

router = APIRouter()

pinecone_index = setup_pinecone_index()
sparse_index = setup_sparse_pinecone_index()
class AskRequest(BaseModel):
    question: str
    person_name: str


@router.post("/upload")
async def upload_pdf(
    person_name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Uploads a PDF, parses it into markdown via LlamaParse,
    structures and chunks it, and ingests it into Pinecone under the person's namespace.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_file_path = tmp.name

    try:
        # 1. Parse markdown
        markdown_text = extract_markdown(temp_file_path)

        # 2. Structure & Chunk
        structure = classify_structure(markdown_text)
        chunks = prepare_chunks(structure)

        if not chunks:
            raise HTTPException(status_code=400, detail="No extractable text or sections found in the PDF.")

        # 3. Ingest into Pinecone under the user's namespace
        ingested_chunks = ingest(
            chunks=chunks,
            person_name=person_name,
            pinecone_index=pinecone_index,
            sparse_index=sparse_index
        )

        return {
            "status": "success",
            "message": f"Successfully ingested {len(ingested_chunks)} chunks for {person_name}.",
            "person_name": person_name,
            "total_chunks": len(ingested_chunks)
        }

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post("/ask")
async def ask_question(payload: AskRequest):
    """
    Asks a question about an ingested document for a specific person.
    """
    try:
        answer = ask(
            user_question=payload.question,
            sparse_index=sparse_index,
            pinecone_index=pinecone_index,
            person_name=payload.person_name
        )
        return {
            "status": "success",
            "question": payload.question,
            "person_name": payload.person_name,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    