import re
import numpy as np
from rank_bm25 import BM25Okapi

from app.embeddings import embed_query


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def bm25_search(query: str, bm25: BM25Okapi, top_k: int = 20) -> list[int]:
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return ranked_indices.tolist()


def semantic_search(query: str, embeddings: np.ndarray, top_k: int = 20) -> list[int]:
    query_vector = embed_query(query)

    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vector)
    norms[norms == 0] = 1e-9
    similarities = (embeddings @ query_vector) / norms

    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return ranked_indices.tolist()


def rrf_fuse(ranked_lists: list[list[int]], k: int = 60, top_k: int = 20) -> list[int]:
    scores: dict[int, float] = {}

    for ranked_list in ranked_lists:
        for rank, chunk_index in enumerate(ranked_list, start=1):
            scores[chunk_index] = scores.get(chunk_index, 0) + 1 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_index for chunk_index, _ in fused[:top_k]]


def hybrid_search(query: str, chunks: list[dict], embeddings: np.ndarray, bm25: BM25Okapi, top_k: int = 20) -> list[int]:
    bm25_results = bm25_search(query, bm25, top_k=top_k)
    semantic_results = semantic_search(query, embeddings, top_k=top_k)
    return rrf_fuse([bm25_results, semantic_results], top_k=top_k)