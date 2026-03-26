"""Streamlit web interface for RAGFromScratch."""

import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


def _truncate(text: str, max_len: int) -> str:
    """Return *text* truncated to *max_len* characters with an ellipsis."""
    return text[:max_len] + "…" if len(text) > max_len else text

st.set_page_config(page_title="RAGFromScratch", page_icon="🔍", layout="wide")
st.title("🔍 RAGFromScratch — Retrieval-Augmented Generation")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_base = st.text_input("FastAPI base URL", value=API_BASE)
    st.markdown("---")
    st.markdown("**Tabs**")
    st.markdown("- 📤 Upload Document\n- 💬 Chat\n- 📚 Sources\n- 📊 Monitor")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_upload, tab_chat, tab_sources, tab_monitor = st.tabs(
    ["📤 Upload Document", "💬 Chat", "📚 Sources", "📊 Monitor"]
)

# ─── 1. Upload Document ──────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload a Q&A Text Document")
    st.markdown(
        "Upload a `.txt` file with Q&A pairs in the format:\n"
        "```\nQ: What is RAG?\nA: RAG stands for ...\n```"
    )
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded_file is not None:
        if st.button("📤 Upload & Index"):
            with st.spinner("Uploading and indexing…"):
                try:
                    response = requests.post(
                        f"{api_base}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")},
                        timeout=60,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(
                            f"✅ {data['message']} ({data['chunks_indexed']} chunks indexed)"
                        )
                    else:
                        st.error(f"❌ Error: {response.json().get('detail', response.text)}")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to the API at {api_base}. Is the server running?")

# ─── 2. Chat ─────────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("💬 Ask a Question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Your question:", placeholder="What is RAG?")
    top_k = st.slider("Top-K documents to retrieve", min_value=1, max_value=10, value=4)

    if st.button("🚀 Ask") and question:
        with st.spinner("Thinking…"):
            try:
                response = requests.post(
                    f"{api_base}/chat",
                    json={"question": question, "top_k": top_k},
                    timeout=120,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.chat_history.append(data)
                    st.success("✅ Answer received!")
                else:
                    st.error(f"❌ Error: {response.json().get('detail', response.text)}")
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to the API at {api_base}. Is the server running?")

    for entry in reversed(st.session_state.chat_history):
        with st.expander(f"Q: {entry['question']}", expanded=True):
            st.markdown(f"**Answer:** {entry['answer']}")

            if entry.get("sources"):
                st.markdown("**📚 Retrieved Sources:**")
                for src in entry["sources"]:
                    st.markdown(f"- `{src}`")

            scores = entry.get("scores", {})
            if scores:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Relevance", f"{scores.get('relevance_score', 0):.2f}")
                col2.metric("Faithfulness", f"{scores.get('faithfulness_score', 0):.2f}")
                col3.metric("Completeness", f"{scores.get('completeness_score', 0):.2f}")
                col4.metric("Latency (ms)", f"{scores.get('latency_ms', 0):.0f}")

# ─── 3. Sources ───────────────────────────────────────────────────────────────
with tab_sources:
    st.subheader("📚 Indexed Source Documents")
    if st.button("🔄 Refresh Sources"):
        try:
            response = requests.get(f"{api_base}/sources", timeout=30)
            if response.status_code == 200:
                sources = response.json()
                if sources:
                    for doc in sources:
                        with st.expander(
                            f"[{doc['filename']}] Chunk #{doc['chunk_index']}",
                            expanded=False,
                        ):
                            st.text(doc["content"])
                            st.caption(f"Indexed at: {doc.get('created_at', 'N/A')}")
                else:
                    st.info("No documents indexed yet. Upload a document first.")
            else:
                st.error(f"❌ Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to the API at {api_base}. Is the server running?")

# ─── 4. Monitor ───────────────────────────────────────────────────────────────
with tab_monitor:
    st.subheader("📊 Request / Response Monitor")
    if st.button("🔄 Refresh History"):
        try:
            response = requests.get(f"{api_base}/history", timeout=30)
            if response.status_code == 200:
                history = response.json()
                if history:
                    import pandas as pd

                    rows = []
                    for r in history:
                        rows.append(
                            {
                                "ID": r["id"],
                                "Question": _truncate(r["question"], 80),
                                "Answer": _truncate(r["answer"], 120),
                                "Sources": ", ".join(r.get("sources", [])),
                                "Relevance": f"{r.get('relevance_score', 0):.2f}",
                                "Faithfulness": f"{r.get('faithfulness_score', 0):.2f}",
                                "Completeness": f"{r.get('completeness_score', 0):.2f}",
                                "Latency (ms)": f"{r.get('latency_ms', 0):.0f}",
                                "Created At": r.get("created_at", ""),
                            }
                        )
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)

                    # Score trend chart
                    st.markdown("### Score Trends")
                    chart_data = pd.DataFrame(
                        {
                            "Relevance": [float(row["Relevance"]) for row in rows],
                            "Faithfulness": [float(row["Faithfulness"]) for row in rows],
                            "Completeness": [float(row["Completeness"]) for row in rows],
                        }
                    )
                    st.line_chart(chart_data)
                else:
                    st.info("No history yet. Ask a question in the Chat tab first.")
            else:
                st.error(f"❌ Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to the API at {api_base}. Is the server running?")
