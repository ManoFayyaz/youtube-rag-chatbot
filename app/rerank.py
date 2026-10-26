import os
import cohere
from dotenv import load_dotenv

load_dotenv()

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


def rerank_chunks(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    texts = [c["text"] for c in chunks]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_n,
    )

    reranked = []
    for result in response.results:
        chunk = chunks[result.index]
        reranked.append({
            "text": chunk["text"],
            "start": chunk["start"],
            "relevance_score": result.relevance_score,
        })

    return reranked