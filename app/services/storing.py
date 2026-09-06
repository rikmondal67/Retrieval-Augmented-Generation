from pinecone import ServerlessSpec
from app.core.config import INDEX_NAME, SPARSE_INDEX_NAME, EMBEDDING_MODEL
from app.clients.pinecone_client import pc
from app.clients.voyage_client import vo
from app.services.embedding import embed_sparse
from app.services.chunking import make_chunk_id

def setup_pinecone_index():
    """Dense semantic index (Voyage embeddings)."""
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME, dimension=1024, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)

def setup_sparse_pinecone_index():
    if SPARSE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=SPARSE_INDEX_NAME,
            vector_type="sparse",
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(SPARSE_INDEX_NAME)

def ingest(chunks, username, pinecone_index, sparse_index, batch_size=20):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["embedding_text"] for c in batch]

        # Dense vectors (Voyage)
        dense_vectors = vo.embed(texts=texts, model=EMBEDDING_MODEL, input_type="document").embeddings

        # Sparse vectors (Pinecone hosted API)
        sparse_vectors = embed_sparse(texts, input_type="passage")

        dense_records = []
        sparse_records = []
        for c, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
            chunk_id = make_chunk_id(username, c)
            metadata = {"text": c["raw_text"], "section": c["section"], "title": c["title"], "file_name": username}

            dense_records.append({"id": chunk_id, "values": dense_vec, "metadata": metadata})
            sparse_records.append({"id": chunk_id, "sparse_values": sparse_vec, "metadata": metadata})

        # Notice the addition of namespace=username here for user isolation
        pinecone_index.upsert(vectors=dense_records, namespace=username)
        sparse_index.upsert(vectors=sparse_records, namespace=username)
        print(f"Stored chunks {i + 1}-{min(i + batch_size, len(chunks))} of {len(chunks)} for {username}")
        
    return chunks