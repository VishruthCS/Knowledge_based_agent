# app.py
import os
import streamlit as st
from ingest import process_and_index, query_index, get_embeddings

UPLOAD_DIR = "uploads"
PERSIST_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Knowledge Base Agent (Gemini + Fallback)", layout="wide")
st.title("📘 Knowledge Base Agent — Gemini (with local fallback)")

# -------------------------
# Session state
# -------------------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "index_status" not in st.session_state:
    st.session_state.index_status = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embedding_backend" not in st.session_state:
    # will store a short string like "Gemini" or "LocalMiniLM"
    st.session_state.embedding_backend = None
if "embedding_checked" not in st.session_state:
    st.session_state.embedding_checked = False

# -------------------------
# Helper: detect embedding backend (cached)
# -------------------------
def detect_embedding_backend():
    """
    Calls ingest.get_embeddings() once to learn which embedding backend will be used.
    This may trigger a small smoke-test: Gemini embedding attempt or local model load.
    We cache the result in session_state.embedding_backend to avoid repeated work.
    """
    if st.session_state.embedding_checked and st.session_state.embedding_backend:
        return st.session_state.embedding_backend

    try:
        emb = get_embeddings()
        # try to infer from type name / module
        name = type(emb).__name__
        module = type(emb).__module__
        if "GoogleGenerativeAIEmbeddings" in name or "google" in module.lower():
            backend = "Gemini (Cloud embeddings)"
        elif "Local" in name or "Sentence" in name or "MiniLM" in name or "sentence_transformers" in module.lower():
            backend = "Local (sentence-transformers: all-MiniLM-L6-v2)"
        else:
            backend = f"Other ({name})"
    except Exception as e:
        # if detection failed, mark as unknown and include error
        backend = f"Unavailable (error: {e})"

    st.session_state.embedding_backend = backend
    st.session_state.embedding_checked = True
    return backend

# -------------------------
# UI: Upload
# -------------------------
st.header("📁 Upload Documents")
uploaded = st.file_uploader("Upload PDF / DOCX / TXT (multiple)", accept_multiple_files=True)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Save uploaded files"):
        if uploaded:
            saved = 0
            for f in uploaded:
                path = os.path.join(UPLOAD_DIR, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                if path not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(path)
                saved += 1
            st.success(f"Saved {saved} file(s).")
            st.session_state.indexed = False
            st.session_state.index_status = ""
            # reset embedding check so detection will re-run if desired
            st.session_state.embedding_checked = False
            st.session_state.embedding_backend = None
        else:
            st.warning("No files selected to save.")

with col2:
    if st.button("Clear uploaded files"):
        # try removing files from disk as well
        for p in list(st.session_state.uploaded_files):
            try:
                os.remove(p)
            except Exception:
                pass
        st.session_state.uploaded_files = []
        st.session_state.chat_history = []
        st.session_state.indexed = False
        st.session_state.index_status = ""
        st.session_state.embedding_checked = False
        st.session_state.embedding_backend = None
        st.success("Cleared uploads and session state.")

if st.session_state.uploaded_files:
    st.write("Saved files:")
    for p in st.session_state.uploaded_files:
        st.write("•", os.path.basename(p))
else:
    st.info("No saved files yet. Upload and hit 'Save uploaded files'.")

# -------------------------
# Embedding backend indicator (detect on demand)
# -------------------------
st.markdown("---")
st.header("⚙️ Embedding Backend")
colA, colB = st.columns([3, 1])

# LEFT SIDE
with colA:
    st.write("Detect which embeddings will be used for indexing (Gemini or Local fallback).")
    if st.button("Check embedding backend"):
        with st.spinner("Detecting embedding backend..."):
            backend = detect_embedding_backend()
            st.success(f"Embedding backend: {backend}")

# RIGHT SIDE — Show cached backend info (NO else here)
with colB:
    if st.session_state.embedding_backend:
        st.info(f"Current embedding backend: **{st.session_state.embedding_backend}**")
    else:
        st.write("Embedding backend not checked yet.")

# Show fallback banner if local
if st.session_state.embedding_backend and "Local" in st.session_state.embedding_backend:
    st.warning(
        f"⚠️ Using Local Embeddings: {st.session_state.embedding_backend}\n"
        "Gemini embedding quota unavailable. Fallback active."
    )

# -------------------------
# INDEX / INGEST
# -------------------------
st.markdown("---")
st.header("🔍 Build Knowledge Index")

col_idx1, col_idx2 = st.columns([1, 1])
with col_idx1:
    if st.button("Index uploaded files"):
        if not st.session_state.uploaded_files:
            st.error("No uploaded files to index. Upload and save files first.")
        else:
            # Run detection once before indexing so user sees banner if local fallback will be used
            if not st.session_state.embedding_checked:
                with st.spinner("Detecting embedding backend..."):
                    detect_embedding_backend()

            st.session_state.index_status = "Indexing..."
            st.session_state.indexed = False
            with st.spinner("Indexing documents (this may take a while)..."):
                res = process_and_index(st.session_state.uploaded_files, persist_directory=PERSIST_DIR)
                st.session_state.indexed = res.get("success", False)
                st.session_state.index_status = res.get("message", "")
                if st.session_state.indexed:
                    st.success("Indexing complete.")
                else:
                    st.error("Indexing failed: " + st.session_state.index_status)

with col_idx2:
    if st.button("Force re-index (clear existing)"):
        import shutil
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        st.session_state.indexed = False
        st.session_state.index_status = ""
        st.success("Cleared persisted index. Click 'Index uploaded files' to rebuild.")

if st.session_state.index_status:
    st.write("Index status:", st.session_state.index_status)

# -------------------------
# QUERY / CHAT
# -------------------------
st.markdown("---")
st.header("💬 Ask Questions")
question = st.text_input("Enter your question")

colq1, colq2 = st.columns([1, 1])
with colq1:
    if st.button("Get Answer"):
        if not question:
            st.warning("Please type a question first.")
        elif not st.session_state.indexed:
            st.warning("Index is not built. Index uploaded files first for best results.")
        else:
            with st.spinner("Querying..."):
                out = query_index(question, persist_directory=PERSIST_DIR, top_k=5)
                ans = out.get("answer", "")
                sources = out.get("sources", [])
                st.session_state.chat_history.insert(0, (question, ans, sources))
                st.experimental_rerun()

with colq2:
    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.success("Cleared chat history.")

# Show chat history
st.markdown("### Chat history")
if st.session_state.chat_history:
    for i, (q, a, sources) in enumerate(st.session_state.chat_history):
        exp = st.expander(f"Q: {q}", expanded=(i == 0))
        with exp:
            st.write("**A:**", a)
            if sources:
                st.write("**Sources:**")
                for s in sources:
                    fn = s.get("filename") or s.get("source") or "unknown"
                    page = s.get("page", "")
                    snippet = (s.get("text") or "")[:400].replace("\n", " ")
                    st.write(f"- {fn} (page: {page}) — _{snippet}_")
                    orig = s.get("file_path")
                    if orig and os.path.exists(orig):
                        try:
                            st.download_button(label=f"Download {os.path.basename(orig)}",
                                               data=open(orig, "rb"),
                                               file_name=os.path.basename(orig))
                        except Exception:
                            pass
else:
    st.info("No questions asked yet. Ask one above!")

st.markdown("---")
st.caption("Built for the AI Agent Development Challenge — shows whether local embeddings are used.")
