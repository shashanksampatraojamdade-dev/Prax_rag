"""Prax RAG — Streamlit chatbot with knowledge-base management.

Run:
    PRAX_API_URL=http://localhost:8000/api/v1 \\
    .venv/bin/streamlit run ui/app.py
"""

import requests
import streamlit as st

import api_client as api

st.set_page_config(page_title="Prax RAG Chatbot", page_icon="🤖", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def api_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            return str(exc.response.json().get("detail", exc))
        except Exception:
            return exc.response.text or str(exc)
    if isinstance(exc, requests.ConnectionError):
        return "Cannot reach the API. Is the backend running on the configured URL?"
    return str(exc)


def format_source(s: dict) -> str:
    """Render one citation: title, path, upload date, version, score, preview."""
    title = s["title"] or s["source"]
    uploaded = s.get("uploaded_at")
    when = str(uploaded).replace("T", " ")[:19] if uploaded else "unknown date"
    version = s.get("version", 1)
    return (
        f"**[{s['rank']}] {title}** · score `{s['score']}`\n\n"
        f"- 📄 Path: `{s['source']}`\n"
        f"- 🕒 Uploaded: `{when}`  ·  version `v{version}`\n\n"
        f"> {s['preview']}"
    )


def stream_answer(question: str, top_k: int):
    """Generator for st.write_stream; captures sources & timings into session."""
    st.session_state.pending_sources = []
    st.session_state.pending_timings = None
    st.session_state.pending_confidence = None
    st.session_state.pending_mode = None
    for event in api.query_stream(question, top_k=top_k):
        etype = event.get("type")
        if etype == "sources":
            st.session_state.pending_sources = event.get("sources", [])
            st.session_state.pending_mode = event.get("retrieval_mode")
            if event.get("confidence") is not None:
                st.session_state.pending_confidence = (
                    event.get("confidence"),
                    event.get("confidence_label", ""),
                )
        elif etype == "token":
            yield event.get("text", "")
        elif etype == "done":
            st.session_state.pending_timings = event.get("timings")
            if event.get("retrieval_mode"):
                st.session_state.pending_mode = event.get("retrieval_mode")
            if event.get("confidence") is not None:
                st.session_state.pending_confidence = (
                    event.get("confidence"),
                    event.get("confidence_label", ""),
                )
            return


def confidence_caption(c: tuple | None, mode: str | None = None) -> str:
    if not c:
        return ""
    score, label = c
    emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(label, "⚪️")
    mode_txt = f" · 🔎 {mode} retrieval" if mode else ""
    return f"{emoji} Confidence: **{score:.0%}** ({label}){mode_txt}"


def timing_caption(t: dict | None) -> str:
    if not t:
        return ""
    return (
        f"⏱️ retrieval {t['retrieval_ms']:.0f} ms · "
        f"generation {t['generation_ms']:.0f} ms · "
        f"total {t['total_ms']:.0f} ms"
    )


# --------------------------------------------------------------------------- #
# Sidebar — Knowledge Base management
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📚 Knowledge Base")

    # Connection status
    try:
        h = api.health()
        st.success(f"API online · {h['documents']} docs · {h['chunks']} chunks")
    except Exception as exc:
        st.error(api_error(exc))

    st.divider()
    st.subheader("Ingest a file")
    uploaded = st.file_uploader(
        "PDF, DOCX, or text/markdown",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    file_title = st.text_input("Title (optional)", key="file_title")
    if st.button("Ingest file(s)", use_container_width=True, disabled=not uploaded):
        for uf in uploaded:
            try:
                res = api.ingest_file(uf.name, uf.getvalue(), file_title or None)
                st.success(f"{uf.name}: {res['chunks_ingested']} chunk(s)")
            except Exception as exc:
                st.error(f"{uf.name}: {api_error(exc)}")
        st.rerun()

    st.divider()
    st.subheader("Ingest text")
    raw_text = st.text_area("Paste text", height=120, key="raw_text")
    raw_source = st.text_input("Source id", value="pasted-note", key="raw_source")
    if st.button("Ingest text", use_container_width=True, disabled=not raw_text.strip()):
        try:
            res = api.ingest_text(raw_text, raw_source, None)
            st.success(f"Ingested {res['chunks_ingested']} chunk(s)")
            st.rerun()
        except Exception as exc:
            st.error(api_error(exc))

    st.divider()
    st.subheader("Indexed documents")
    try:
        docs = api.list_documents()
        st.caption(f"{docs['total']} document(s)")
        for d in docs["documents"]:
            col1, col2 = st.columns([0.8, 0.2])
            label = d["title"] or d["source"]
            col1.write(f"**{label}**  \n`{d['source']}` · {d['chunk_count']} chunk(s)")
            if col2.button("🗑️", key=f"del_{d['id']}", help="Delete"):
                api.delete_document(d["id"])
                st.rerun()
    except Exception as exc:
        st.error(api_error(exc))

    st.divider()
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 5)
    if st.button("⚠️ Reset entire KB", use_container_width=True):
        try:
            api.reset()
            st.session_state.messages = []
            st.rerun()
        except Exception as exc:
            st.error(api_error(exc))


# --------------------------------------------------------------------------- #
# Main — Chat window
# --------------------------------------------------------------------------- #
st.title("🤖 Prax RAG Chatbot")
st.caption("Grounded answers from your knowledge base · powered by Vertex AI + pgvector")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("confidence"):
            st.caption(confidence_caption(msg["confidence"], msg.get("mode")))
        if msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(format_source(s))

# Chat input
if prompt := st.chat_input("Ask a question about your knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(prompt, top_k))
            confidence = st.session_state.get("pending_confidence")
            mode = st.session_state.get("pending_mode")
            if confidence:
                st.caption(confidence_caption(confidence, mode))
            timings = st.session_state.get("pending_timings")
            if timings:
                st.caption(timing_caption(timings))
            sources = st.session_state.get("pending_sources", [])
            if sources:
                with st.expander(f"📎 Sources ({len(sources)})"):
                    for s in sources:
                        st.markdown(format_source(s))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "confidence": confidence,
                    "mode": mode,
                    "sources": sources,
                    "timings": timings,
                }
            )
        except Exception as exc:
            st.error(api_error(exc))
