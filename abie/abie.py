"""
ABIE — Adaptive Business Intelligence Engine.
Main Streamlit app for multi-document RAG-powered Q&A.
"""

import os
import sys
from pathlib import Path

# Ensure parent directory is on path so "abie" package resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from abie.utils import embedder, loader, qa_chain

# Load .env from the abie package directory
load_dotenv(Path(__file__).resolve().parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def init_session_state() -> None:
    """Initialize or ensure all required keys exist in st.session_state."""
    defaults = {
        "uploaded_docs": [],
        "vectorstore": None,
        "doc_chunk_stats": {},
        "chat_history": [],
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.2,
        "selected_docs": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_sidebar() -> None:
    """Render the left sidebar with upload, analytics, and controls."""
    st.sidebar.markdown("## 🧠 ABIE")
    st.sidebar.markdown("*Adaptive Business Intelligence Engine*")
    st.sidebar.divider()

    uploaded = st.sidebar.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        embedding_model = embedder.create_embedding_model()
        vs = st.session_state.get("vectorstore")
        if vs is None:
            vs = embedder.get_or_create_vectorstore(embedding_model)
            st.session_state["vectorstore"] = vs

        page_entries, failed_files = loader.extract_pages_from_pdfs(uploaded)
        for fail in failed_files:
            st.sidebar.warning(
                f"PDF parsing failed: **{fail['file_name']}** — {fail['reason']}"
            )

        if page_entries:
            docs, per_doc = loader.chunk_documents(
                page_entries, chunk_size=800, overlap=100
            )
            embedder.add_documents_to_vectorstore(vs, docs)

            file_page_counts: dict[str, int] = {}
            for e in page_entries:
                fn = e["file_name"]
                pn = e["page_number"]
                file_page_counts[fn] = max(file_page_counts.get(fn, 0), pn)

            for fn, pc in file_page_counts.items():
                existing = next(
                    (d for d in st.session_state["uploaded_docs"] if d["file_name"] == fn),
                    None,
                )
                if not existing:
                    st.session_state["uploaded_docs"].append(
                        {"file_name": fn, "page_count": pc}
                    )

            for fn, cnt in per_doc.items():
                st.session_state["doc_chunk_stats"][fn] = (
                    st.session_state["doc_chunk_stats"].get(fn, 0) + cnt
                )

        st.sidebar.success(f"Processed {len(page_entries)} pages from {len(uploaded)} file(s).")
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### Document Analytics")

    total_docs = len(st.session_state["uploaded_docs"])
    total_chunks = sum(st.session_state["doc_chunk_stats"].values())
    st.sidebar.metric("Total documents uploaded", total_docs)
    st.sidebar.metric("Total chunks stored", total_chunks)

    if st.session_state["doc_chunk_stats"]:
        df = pd.DataFrame(
            list(st.session_state["doc_chunk_stats"].items()),
            columns=["Document", "Chunks"],
        )
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(df["Document"], df["Chunks"], color="steelblue", alpha=0.8)
        ax.set_xlabel("Chunk count")
        ax.set_title("Chunks per document")
        plt.tight_layout()
        st.sidebar.pyplot(fig)
        plt.close()

    st.sidebar.divider()
    st.sidebar.markdown("### Model settings")
    st.sidebar.text(f"Model: {st.session_state['model_name']}")
    st.session_state["temperature"] = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["temperature"],
        step=0.1,
    )

    st.sidebar.divider()
    if st.sidebar.button("Clear History"):
        st.session_state["chat_history"] = []
        st.rerun()

    if st.sidebar.button("Clear & Reset"):
        st.session_state["chat_history"] = []
        st.session_state["uploaded_docs"] = []
        st.session_state["doc_chunk_stats"] = {}
        st.session_state["vectorstore"] = None
        st.session_state["selected_docs"] = []
        embedder.reset_chroma_collection()
        st.rerun()


def render_main() -> None:
    """Render the main content area with query input, answer, and history."""
    st.title("🧠 ABIE — Adaptive Business Intelligence Engine")
    st.markdown("*Multi-document RAG-powered Q&A for business insights*")
    st.divider()

    has_docs = bool(st.session_state["doc_chunk_stats"])
    vs = st.session_state.get("vectorstore")

    if st.session_state["uploaded_docs"]:
        st.markdown("#### Select documents to query")
        selected = []
        for doc in st.session_state["uploaded_docs"]:
            fn = doc["file_name"]
            pc = doc["page_count"]
            if st.checkbox(f"{fn} ({pc} pages)", key=f"sel_{fn}", value=True):
                selected.append(fn)
        st.session_state["selected_docs"] = selected

    if not has_docs or vs is None:
        st.text_input(
            "Ask a question",
            key="query_input",
            placeholder="Upload a document first",
            disabled=True,
        )
        st.info("Upload one or more PDF documents in the sidebar to enable Q&A.")
        return

    query = st.text_input("Ask a question", key="query_input", placeholder="Enter your business question...")

    if st.button("Ask ABIE"):
        if not query or not query.strip():
            st.warning("Please enter a question")
            return

        selected = st.session_state.get("selected_docs") or [
            d["file_name"] for d in st.session_state["uploaded_docs"]
        ]
        filter_sources = selected if selected else None

        with st.spinner("Analyzing your question against uploaded documents..."):
            try:
                chunks_with_scores = embedder.get_relevant_chunks(
                    vs, query.strip(), k=4, filter_sources=filter_sources
                )
            except Exception as e:
                st.error(f"Retrieval error: {e}. Please try again.")
                return

            if not chunks_with_scores:
                st.warning("No relevant chunks found. Try rephrasing or uploading more documents.")
                return

            docs_only = [d for d, _ in chunks_with_scores]
            scores = [s for _, s in chunks_with_scores]

            try:
                answer = qa_chain.run_business_qa(
                    query.strip(),
                    docs_only,
                    st.session_state["model_name"],
                    st.session_state["temperature"],
                )
            except Exception as e:
                st.error(f"OpenAI API error: {str(e)}. Please check your API key and try again.")
                return

            label, avg_score = qa_chain.compute_confidence_label(scores)
            sources = [
                f"{d.metadata.get('source', 'Unknown')} — page {d.metadata.get('page', '?')}"
                for d in docs_only
            ]

            record = {
                "question": query.strip(),
                "answer": answer,
                "sources": sources,
                "confidence_label": label,
                "confidence_score": avg_score,
            }
            st.session_state["chat_history"] = (
                st.session_state["chat_history"] + [record]
            )[-5:]

        st.success("Answer generated")

    if st.session_state["chat_history"]:
        latest = st.session_state["chat_history"][-1]
        st.markdown("#### Answer")
        st.markdown(
            f'<div style="background:#1e1e1e;padding:1rem;border-radius:8px;border:1px solid #333;">'
            f'{latest["answer"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Confidence:** {latest['confidence_label']} (score: {latest['confidence_score']:.2f})")
        st.markdown("**Sources:**")
        for s in latest["sources"]:
            st.markdown(f"- {s}")

    st.divider()
    with st.expander("Conversation History (last 5)"):
        for i, rec in enumerate(st.session_state["chat_history"]):
            st.markdown(f"**Q:** {rec['question']}")
            st.markdown(f"**A:** {rec['answer'][:200]}{'...' if len(rec['answer']) > 200 else ''}")
            st.markdown(f"*{rec['confidence_label']}*")
            st.divider()


def main() -> None:
    """Entry point: configure page, check API key, and render the app."""
    st.set_page_config(
        page_title="🧠 ABIE — Adaptive Business Intelligence Engine",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; }
        div[data-testid="stSidebar"] { background-color: #1a1a2e; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
        st.error(
            "**Google API key not found.** Add `GOOGLE_API_KEY=your_google_api_key_here` to "
            "`.env` in the abie folder and replace with your actual key. "
            "Get a key at https://aistudio.google.com/apikey"
        )
        return

    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
