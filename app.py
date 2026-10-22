"""Streamlit chat UI: upload PDFs, ask questions, see cited sources.

Run:  streamlit run app.py
"""
from __future__ import annotations

import itertools
import re
import time

import chromadb
import streamlit as st

from rag import RAGPipeline, Settings
from rag.generator import cited_indices
from rag.store import EmbeddingMismatchError

st.set_page_config(page_title="DocRAG · Chat with your PDFs", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(Settings())


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("📄 DocRAG")
    st.caption("Hybrid retrieval (dense + BM25) · cited answers")

    try:
        rag = get_pipeline()
    except EmbeddingMismatchError as e:
        st.error(str(e))
        if st.button("Clear the index and start fresh", type="primary", use_container_width=True):
            s = Settings()
            chromadb.PersistentClient(path=str(s.data_dir / "chroma")).delete_collection(s.collection)
            st.cache_resource.clear()
            st.rerun()
        st.stop()

    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Index uploaded files", type="primary", use_container_width=True):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    r = rag.ingest_pdf(f.getvalue(), f.name)
                    if r.skipped:
                        st.toast(f"{f.name} is already indexed")
                    else:
                        st.toast(f"{f.name}: {r.pages} pages → {r.chunks} chunks")
                except Exception as e:  # show the reason, keep going with the rest
                    st.error(f"{f.name}: {e}")

    docs = rag.list_docs()
    st.subheader(f"Library ({len(docs)})")
    selected: list[str] = []
    for d in docs:
        c1, c2 = st.columns([5, 1])
        if c1.checkbox(f"{d['source']}  ·  {d['pages']}p", value=True, key=f"sel_{d['doc_id']}"):
            selected.append(d["doc_id"])
        if c2.button("🗑", key=f"del_{d['doc_id']}", help="Remove from index"):
            rag.delete_doc(d["doc_id"])
            st.rerun()

    with st.expander("Retrieval settings"):
        mode = st.radio("Retriever", ["hybrid", "dense", "bm25"], horizontal=True)
        top_k = st.slider("Chunks sent to the LLM", 1, 10, rag.s.top_k)
        rerank = st.toggle("Cross-encoder rerank", value=rag.s.use_reranker,
                           help="Needs `pip install sentence-transformers`")
        show_debug = st.toggle("Show search query", value=False)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- chat ----------------
def render_sources(sources: list[dict], cited: list[int]) -> None:
    if not sources:
        return
    order = cited or list(range(1, len(sources) + 1))
    label = f"Sources ({len(cited)} cited)" if cited else "Retrieved passages"
    with st.expander(label):
        for i in order:
            s = sources[i - 1]
            st.markdown(f"**[{i}] {s['source']} — page {s['page']}**")
            st.caption(s["text"][:600] + ("…" if len(s["text"]) > 600 else ""))


st.session_state.setdefault("messages", [])

if not docs:
    st.markdown("### Upload a PDF in the sidebar to get started")
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []), m.get("cited", []))

if question := st.chat_input("Ask something about your documents"):
    if not selected:
        st.warning("Select at least one document in the sidebar.")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        started = time.perf_counter()
        # A live status box with a spinner: each step is listed as it starts, and the
        # spinner keeps going until the model's first words arrive (that wait can take
        # several seconds on a local model, especially the first time it loads).
        status = st.status("Working on it…", expanded=True)
        try:
            with status:
                query = question
                if rag.needs_rewrite(question, history):
                    st.write("✏️ Understanding your follow-up question…")
                    query = rag.rewrite_query(question, history)
                    if show_debug:
                        st.caption(f"Search query: _{query}_")

                st.write("🔎 Searching your documents…")
                if rerank:
                    st.write("🎯 Re-ranking the best matches… (the first time downloads a ~90 MB model)")
                hits = rag.search(query, doc_ids=selected, top_k=top_k, mode=mode, rerank=rerank)
                st.write(f"📄 Found {len(hits)} relevant passage{'s' if len(hits) != 1 else ''}")

                st.write("🧠 Writing the answer… (a local model can take a few seconds to start)")
                # Give the model the standalone question too, so "that"/"it" are already resolved.
                stream = rag.generate_stream(query, hits, history)
                first = next(stream, "")  # spinner stays on until the first words arrive

            found = f"Found {len(hits)} passage{'s' if len(hits) != 1 else ''}"
            status.update(label=f"{found} · writing…", state="complete", expanded=False)
            text = st.write_stream(itertools.chain([first], stream))
            status.update(label=f"{found} · answered in {time.perf_counter() - started:.1f}s")
        except ImportError:
            status.update(label="Reranker not installed", state="error")
            st.error("Reranking needs `pip install sentence-transformers`. Turn it off or install it.")
            st.stop()
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"Something went wrong: {e}")
            st.stop()

        sources = [{"source": h.chunk.source, "page": h.chunk.page, "text": h.chunk.text} for h in hits]
        cited = cited_indices(text, len(sources))
        render_sources(sources, cited)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources, "cited": cited}
    )
