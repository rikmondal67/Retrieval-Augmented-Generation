import os
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("pinecone_api")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

EMBEDDING_MODEL = "voyage-4-large"
RERANK_MODEL = "rerank-2"
INDEX_NAME = "cvcopy-index"

SPARSE_MODEL = "pinecone-sparse-english-v0"
SPARSE_INDEX_NAME = "cvcopy-sparse-index"