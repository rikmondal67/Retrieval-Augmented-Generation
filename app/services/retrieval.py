from app.clients.voyage_client import vo
from app.core.config import EMBEDDING_MODEL
from app.services.embedding import embed_sparse

def hybrid_retrieve(user_question, sparse_index, pinecone_index, person_name: str, top_k=20):
    # Dense (semantic) side
    query_vector = vo.embed(texts=[user_question], model=EMBEDDING_MODEL, input_type="query").embeddings[0]
    dense_matches = pinecone_index.query(
        vector=query_vector, 
        top_k=top_k, 
        include_metadata=True,
        namespace=person_name
    ).matches
    sparse_query_vector = embed_sparse([user_question], input_type="query")[0]
    sparse_matches = sparse_index.query(
        sparse_vector=sparse_query_vector, 
        top_k=top_k, 
        include_metadata=True,
        namespace=person_name
    ).matches

    merged = {}
    for m in dense_matches:
        metadata = m.metadata or {}
        title = metadata.get("title", "Untitled Section")
        text = metadata.get("text", "")
        merged[title] = text

    for m in sparse_matches:
        metadata = m.metadata or {}
        title = metadata.get("title", "Untitled Section")
        text = metadata.get("text", "")
        if title not in merged:
            merged[title] = text

    return merged