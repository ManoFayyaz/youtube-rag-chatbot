import os
from openai import OpenAI
from dotenv import load_dotenv

from app.chunking import format_timestamp

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a YouTube video using only the transcript excerpts provided to you.

Rules:
- Answer only using the information in the given excerpts.
- If the excerpts don't contain the answer, say "That isn't covered in the video."
- When you use information from an excerpt, mention its timestamp in [MM:SS] format.
- Do not make up information that isn't in the excerpts.
- Keep answers concise and directly relevant to the question."""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        timestamp = format_timestamp(chunk["start"])
        parts.append(f"[{timestamp}] {chunk['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)

    user_prompt = f"""Transcript excerpts:

{context}

Question: {question}"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content