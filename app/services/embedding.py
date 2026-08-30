from app.clients.pinecone_client import pc
from app.core.config import SPARSE_MODEL

def embed_sparse(texts, input_type="passage"):
    response = pc.inference.embed(
        model=SPARSE_MODEL,
        inputs=texts,
        parameters={"input_type": input_type, "truncate": "END"},
    )
    return [
        {"indices": item["sparse_indices"], "values": item["sparse_values"]}
        for item in response.data
    ]