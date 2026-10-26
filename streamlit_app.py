import requests
import streamlit as st

BACKEND = "http://127.0.0.1:8000"

st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")
st.title("YouTube RAG Chatbot")

# --- session state so the video survives reruns ---
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "title" not in st.session_state:
    st.session_state.title = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- sidebar: ingest ---
with st.sidebar:
    st.header("Load a video")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Ingest", use_container_width=True):
        if not url.strip():
            st.warning("Paste a YouTube URL first.")
        else:
            with st.spinner("Fetching transcript and building index..."):
                try:
                    r = requests.post(f"{BACKEND}/ingest", json={"url": url}, timeout=120)
                except requests.RequestException as e:
                    st.error(f"Could not reach backend: {e}")
                    r = None

            if r is not None:
                if r.status_code != 200:
                    st.error(f"Ingest failed ({r.status_code}): {r.json().get('detail', r.text)}")
                else:
                    data = r.json()
                    if not data.get("transcript_available", False):
                        st.error(data.get("message", "No transcript available."))
                    else:
                        st.session_state.video_id = data["video_id"]
                        st.session_state.title = data["title"]
                        st.session_state.messages = []
                        st.success(f"Loaded: {data['title']}")

    if st.session_state.video_id:
        st.caption(f"Active video: **{st.session_state.title}**")
        st.caption(f"ID: `{st.session_state.video_id}`")

# --- main pane: chat ---
if not st.session_state.video_id:
    st.info("Ingest a video from the sidebar to start chatting.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask something about the video...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    r = requests.post(
                        f"{BACKEND}/chat",
                        json={
                            "video_id": st.session_state.video_id,
                            "question": question,
                        },
                        timeout=120,
                    )
                    if r.status_code != 200:
                        answer = f"Error ({r.status_code}): {r.json().get('detail', r.text)}"
                        sources = []
                    else:
                        data = r.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])
                except requests.RequestException as e:
                    answer = f"Could not reach backend: {e}"
                    sources = []

            st.write(answer)

            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        ts = s.get("start", 0)
                        mm, ss = divmod(int(ts), 60)
                        st.markdown(f"**[{mm:02d}:{ss:02d}]** {s.get('text', '')[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})