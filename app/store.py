store: dict = {}


def save_video(video_id: str, title: str, chunks: list[dict], embeddings, bm25):
    store[video_id] = {
        "title": title,
        "chunks": chunks,
        "embeddings": embeddings,
        "bm25": bm25,
    }


def get_video(video_id: str):
    return store.get(video_id)


def video_exists(video_id: str) -> bool:
    return video_id in store