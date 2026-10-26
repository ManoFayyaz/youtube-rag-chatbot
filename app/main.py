from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.youtube import extract_video_id, get_transcript, get_title
from app.chunking import chunk_transcript
from app.embeddings import embed_chunks
from . import store
from fastapi import FastAPI, HTTPException
from app.retrieval import build_bm25_index, hybrid_search
from app.rerank import rerank_chunks
from app.llm import generate_answer
from pydantic import BaseModel


app = FastAPI(title="YouTube RAG Chatbot")


class IngestRequest(BaseModel):
    url: str


class ChatRequest(BaseModel):
    video_id: str
    question: str


@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        video_id = extract_video_id(request.url)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like a valid YouTube link.",
        )

    title = get_title(video_id)

    try:
        transcript = get_transcript(video_id)
    except Exception as e:
        return {
            "video_id": video_id,
            "title": title,
            "transcript_available": False,
            "message": "No transcript is available for this video.",
        }

    try:
        chunks = chunk_transcript(transcript)
        embeddings = embed_chunks(chunks)
        bm25 = build_bm25_index(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process transcript: {e}",
        )

    store.save_video(video_id, title, chunks, embeddings, bm25)

    return {
        "video_id": video_id,
        "title": title,
        "transcript_available": True,
        "num_chunks": len(chunks),
        "embedding_shape": list(embeddings.shape),
    }

@app.get("/debug/{video_id}")
def debug_video(video_id: str):
    video = store.get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found in store.")

    return {
        "title": video["title"],
        "num_chunks": len(video["chunks"]),
        "embedding_shape": list(video["embeddings"].shape),
        "first_chunk": video["chunks"][0],
        "first_chunk_embedding_preview": video["embeddings"][0][:5].tolist(),
    }


@app.post("/chat")
def chat(request: ChatRequest):
    video = store.get_video(request.video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found. Ingest it first.")

    top_indices = hybrid_search(
        request.question,
        video["chunks"],
        video["embeddings"],
        video["bm25"],
        top_k=20,
    )

    candidates = [video["chunks"][i] for i in top_indices]
    top_chunks = rerank_chunks(request.question, candidates, top_n=5)
    answer = generate_answer(request.question, top_chunks)

    return {
        "video_id": request.video_id,
        "question": request.question,
        "answer": answer,
        "sources": top_chunks,
    }