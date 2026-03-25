"""Streamlit web interface for the RAGFromScratch application."""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RAGFromScratch", page_icon="🔍", layout="wide")
st.title("🔍 RAGFromScratch — Retrieval-Augmented Generation")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_base = st.text_input("API Base URL", value=API_BASE)
    st.markdown("---")
    st.markdown(
        "**How to use:**\n"
        "1. Upload a Q&A text file\n"
        "2. Ask a question in the Chat tab\n"
        "3. View retrieved sources and monitoring data"
    )

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_upload, tab_chat, tab_sources, tab_monitor = st.tabs(
    ["📤 Upload", "💬 Chat", "📄 Sources", "📊 Monitor"]
)

# ── Upload Tab ───────────────────────────────────────────────────────────────
with tab_upload:
    st.header("Upload Q&A Document")
    st.markdown(
        "Upload a plain-text file containing Q&A pairs in the following format:\n\n"
        "```\n"
        "Q: What is RAG?\n"
        "A: RAG stands for Retrieval-Augmented Generation...\n\n"
        "Q: What is LangChain?\n"
        "A: LangChain is an open-source framework...\n"
        "```"
    )
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded_file is not None:
        if st.button("📤 Index Document"):
            with st.spinner("Uploading and indexing…"):
                try:
                    resp = requests.post(
                        f"{api_base}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")},
                        timeout=60,
                    )
                    if resp.ok:
                        data = resp.json()
                        st.success(data.get("message", "Document indexed successfully!"))
                        st.json(data)
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to the API at {api_base}. Is the FastAPI server running?")

# ── Chat Tab ─────────────────────────────────────────────────────────────────
with tab_chat:
    st.header("Ask a Question")
    top_k = st.slider("Number of retrieved documents (top-k)", min_value=1, max_value=10, value=3)
    question = st.text_area("Your question", placeholder="What is RAG?", height=100)
    if st.button("🚀 Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer…"):
                try:
                    resp = requests.post(
                        f"{api_base}/chat",
                        json={"question": question, "top_k": top_k},
                        timeout=120,
                    )
                    if resp.ok:
                        data = resp.json()
                        st.markdown("### 💡 Answer")
                        st.write(data["answer"])

                        st.markdown("### 📄 Retrieved Sources")
                        if data.get("sources"):
                            for src in data["sources"]:
                                st.markdown(f"- `{src}`")
                        else:
                            st.info("No sources returned.")

                        st.markdown("### 📊 Quality Scores")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Relevance", f"{data['relevance_score']:.2%}")
                        col2.metric("Faithfulness", f"{data['faithfulness_score']:.2%}")
                        col3.metric("Completeness", f"{data['completeness_score']:.2%}")
                        col4.metric("Latency", f"{data['latency_ms']:.0f} ms")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to the API at {api_base}. Is the FastAPI server running?")

# ── Sources Tab ───────────────────────────────────────────────────────────────
with tab_sources:
    st.header("Indexed Document Sources")
    if st.button("🔄 Refresh Sources"):
        try:
            resp = requests.get(f"{api_base}/sources", timeout=10)
            if resp.ok:
                sources = resp.json()
                if sources:
                    st.table(sources)
                else:
                    st.info("No documents indexed yet. Upload a document first.")
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to the API at {api_base}. Is the FastAPI server running?")

# ── Monitor Tab ───────────────────────────────────────────────────────────────
with tab_monitor:
    st.header("Request History & Quality Monitoring")
    limit = st.number_input("Number of recent entries to show", min_value=1, max_value=500, value=20)
    if st.button("🔄 Refresh History"):
        try:
            resp = requests.get(f"{api_base}/history", params={"limit": limit}, timeout=10)
            if resp.ok:
                history = resp.json()
                if history:
                    df = pd.DataFrame(history)
                    # Display scores as percentages for readability
                    for col in ["relevance_score", "faithfulness_score", "completeness_score"]:
                        if col in df.columns:
                            df[col] = df[col].apply(lambda x: f"{x:.2%}" if x is not None else "N/A")
                    if "latency_ms" in df.columns:
                        df["latency_ms"] = df["latency_ms"].apply(
                            lambda x: f"{x:.0f} ms" if x is not None else "N/A"
                        )
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No request history yet. Ask a question first.")
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to the API at {api_base}. Is the FastAPI server running?")
