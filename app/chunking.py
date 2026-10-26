def chunk_transcript(transcript: list[dict], chunk_size: int = 200, overlap: int = 40) -> list[dict]:
    chunks = []
    current_words = []
    current_start = transcript[0]["start"] if transcript else 0

    for segment in transcript:
        words = segment["text"].split()

        if not current_words:
            current_start = segment["start"]

        current_words.extend(words)

        if len(current_words) >= chunk_size:
            chunks.append({
                "text": " ".join(current_words),
                "start": current_start,
            })

            overlap_words = current_words[-overlap:] if overlap > 0 else []
            current_words = overlap_words

    if current_words:
        chunks.append({
            "text": " ".join(current_words),
            "start": current_start,
        })

    return chunks





def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"

