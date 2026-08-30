from app.clients.voyage_client import vo
from app.core.config import RERANK_MODEL

def rerank(user_question, merged: dict, keep_top=6):
    clean_merged = {title: text for title, text in merged.items() if text and text.strip()}

    if not clean_merged:
        return []

    titles = list(clean_merged.keys())
    texts = list(clean_merged.values())

    actual_top_k = min(keep_top, len(texts))

    result = vo.rerank(query=user_question, documents=texts, model=RERANK_MODEL, top_k=actual_top_k)
    return [(titles[r.index], texts[r.index]) for r in result.results]