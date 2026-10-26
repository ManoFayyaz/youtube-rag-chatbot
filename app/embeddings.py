import os
import cohere
import numpy as np
from dotenv import load_dotenv

load_dotenv()

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]

    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"],
    )

    return np.array(response.embeddings.float_)


def embed_query(query: str) -> np.ndarray:
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    )

    return np.array(response.embeddings.float_[0])