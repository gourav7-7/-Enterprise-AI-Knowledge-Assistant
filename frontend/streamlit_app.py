from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from app.core.exception import AppError
from app.rag.chain import ConversationalRAGChain
from app.rag.ingestion import DocIngestor


UPLOAD_DIR = Path("data/uploads")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def _to_lc_history(messages: list[dict]) -> list:
    history: list = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history

def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        seen = set()
        for s in sources:
            tag = (s["source"], s["page"])
            if tag in seen:
                continue
            seen.add(tag)
            page = s["page"]
            page_str = f"p.{page + 1}" if isinstance(page, int) else "p.?"
            st.markdown(f"- **{s['source']}** {page_str} — {s['snippet']}")
 
 
def main() -> None:
    st.set_page_config(page_title="AI Knowledge Assistant", page_icon="📄")
    st.title("📄 AI Knowledge Assistant")
 
    # --- session state (survives reruns) ---
    if "messages" not in st.session_state:
        st.session_state.messages = []          # the conversation buffer
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "doc_name" not in st.session_state:
        st.session_state.doc_name = None
 
    # --- sidebar: upload + ingest ---
    with st.sidebar:
        st.header("Document")
        uploaded = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded is not None and st.button("Ingest", use_container_width=True):
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            saved_path = UPLOAD_DIR / uploaded.name
            saved_path.write_bytes(uploaded.getbuffer())
 
            with st.spinner("Ingesting..."):
                try:
                    ingestor = DocIngestor()
                    ingestor.reset()  # single-doc demo: clean slate each upload
                    summary = ingestor.ingest(saved_path)
                    st.session_state.chain = ConversationalRAGChain()
                    st.session_state.messages = []
                    st.session_state.doc_name = summary["file"]
                except AppError as e:
                    st.error(f"Ingestion failed: {e.message}")
                else:
                    st.success(
                        f"Ingested {summary['file']} "
                        f"({summary['pages']} pages, {summary['chunks']} chunks)"
                    )
 
        if st.session_state.doc_name:
            st.caption(f"Active document: {st.session_state.doc_name}")
 
    # --- main: render conversation so far ---
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                _render_sources(m.get("sources", []))
 
    # --- chat input ---
    prompt = st.chat_input("Ask a question about the document...")
    if prompt:
        if st.session_state.chain is None:
            st.warning("Upload and ingest a PDF first.")
            return
 
        # show + store the user's turn
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
 
        # history is the PRIOR turns (exclude the question we just added)
        history = _to_lc_history(st.session_state.messages[:-1])
 
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.answer(prompt, history)
            st.markdown(result["answer"])
            _render_sources(result["sources"])
 
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )
 
 
if __name__ == "__main__":
    main()