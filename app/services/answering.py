import json
from app.clients.groq_client import groq_client
from app.services.retrieval import hybrid_retrieve
from app.services.reranking import rerank

def answer_with_verification(user_question, retrieved):
    context_text = "\n\n".join(f"[{title}]\n{text}" for title, text in retrieved)
    prompt = f"""Answer the question using ONLY the context below.
Also assess whether the context looks COMPLETE for this question
(e.g. if items are numbered 1,2,3,4 and one number is missing, flag it).

Context:
{context_text}

Question: {user_question}

Respond strictly in valid JSON format matching this schema:
{{"answer": "...", "complete": true/false, "reason": "..."}}
"""

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="openai/gpt-oss-120b",  
        response_format={"type": "json_object"},
        temperature=1.1
    )

    raw = chat_completion.choices[0].message.content.strip()
    if raw.startswith("```json"): raw = raw[7:]
    elif raw.startswith("```"): raw = raw[3:]
    if raw.endswith("```"): raw = raw[:-3]

    return json.loads(raw.strip())


def ask(user_question, sparse_index, pinecone_index, username: str):
    # Pass username so hybrid_retrieve uses the correct Pinecone namespace
    merged = hybrid_retrieve(user_question, sparse_index, pinecone_index, username, top_k=20)
    top_chunks = rerank(user_question, merged, keep_top=6)
    result = answer_with_verification(user_question, top_chunks)

    if not result["complete"]:
        print(f"\n[Debug] Incomplete answer detected ({result['reason']}), retrying wider...")
        merged = hybrid_retrieve(user_question, sparse_index, pinecone_index, username, top_k=40)
        top_chunks = rerank(user_question, merged, keep_top=10)
        result = answer_with_verification(user_question, top_chunks)

    return result["answer"]

