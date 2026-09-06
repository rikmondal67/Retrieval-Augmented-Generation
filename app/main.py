from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(docs_url=None,redoc_url=None)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Document QA RAG API is running"}