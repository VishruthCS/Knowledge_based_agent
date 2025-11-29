import os
import shutil
import tempfile
import streamlit as st
import graphviz
from backend import process_and_index, rewrite_query, retrieve_documents, stream_answer, generate_knowledge_graph

st.set_page_config(page_title="AI Agent V2.1", layout="wide", page_icon="🕸️")

# --- SESSION STATE SETUP ---
# Create a temporary directory unique to this specific browser session
if "temp_persist_dir" not in st.session_state:
    # This creates a fresh folder like /tmp/tmp123abc...
    st.session_state.temp_persist_dir = tempfile.mkdtemp()
    
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "processed_files" not in st.session_state: 
    st.session_state.processed_files = []

# Constants
UPLOAD_DIR = "uploads"
# CRITICAL FIX: Use the session-specific temp folder instead of the shared "chroma_db"
PERSIST_DIR = st.session_state.temp_persist_dir

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Sidebar
with st.sidebar:
    st.title("🤖 Agent V2.1")
    
    model_choice = st.selectbox(
        "AI Model", 
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        index=0
    )
    
    enable_rewrite = st.toggle(
        "Query Rewriting", 
        value=True, 
        help="Automatically rewrites vague questions for better search results."
    )
    
    complexity = st.select_slider(
        "Answer Complexity", 
        options=["Novice", "Standard", "Expert"], 
        value="Standard"
    )
    
    st.divider()
    
    uploaded_files = st.file_uploader("Upload Docs (Current Session Only)", accept_multiple_files=True)
    if st.button("⚡ Process & Index"):
        if uploaded_files:
            with st.spinner("Processing..."):
                saved = []
                for f in uploaded_files:
                    p = os.path.join(UPLOAD_DIR, f.name)
                    # Write the file to the uploads directory
                    with open(p, "wb") as o: 
                        o.write(f.getbuffer())
                    saved.append(p)
                
                # Process files into the Session-Specific DB
                res = process_and_index(saved, PERSIST_DIR)
                if res["success"]:
                    st.success(res["message"])
                    st.session_state.processed_files = saved
                    st.session_state.messages = []
                else: 
                    st.error(res["message"])
        else:
            st.warning("Upload files first!")

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
st.title("🕸️ AI Knowledge Agent + Graph")
st.caption(f"Features: Reasoning • Rewriting • Knowledge Graph • Complexity: **{complexity}**")
st.info(f"🧠 Memory ID: `{os.path.basename(PERSIST_DIR)}` (Files uploaded here will vanish when you close the tab)")

# Display History
for m in st.session_state.messages:
    with st.chat_message(m["role"]): 
        st.markdown(m["content"])

# User Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    with st.chat_message("assistant"):
        msg_box = st.empty()
        
        # 1. Feature 2: Query Rewriting
        query = prompt
        if enable_rewrite:
            with st.status("Thinking...", expanded=False) as status:
                status.write("Analyzing context...")
                hist = [(m["role"], m["content"]) for m in st.session_state.messages]
                query = rewrite_query(prompt, hist, model_choice)
                status.write(f"Optimized Query: {query}")
                status.update(label="Query Optimized", state="complete")
        
        # 2. Feature 1: Retrieval & Evidence
        # Retrieve ONLY from the session-specific folder
        docs = retrieve_documents(query, PERSIST_DIR)
        
        if docs:
            # TABS: Switch between Text Evidence and Visual Graph
            tab1, tab2 = st.tabs(["📄 Reasoning Trace", "🕸️ Knowledge Graph"])
            
            with tab1:
                st.caption("The agent used these document snippets to answer:")
                for i, d in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}** | Source: *{d.metadata.get('source')}* (Pg {d.metadata.get('page')})")
                    st.caption(d.page_content[:300].replace("\n", " ") + "...")
                    st.divider()
            
            with tab2:
                # Feature 4: Knowledge Graph Generation
                with st.spinner("Generating graph from context..."):
                    dot_code = generate_knowledge_graph(docs, model_choice)
                    try:
                        st.graphviz_chart(dot_code)
                    except Exception as e:
                        st.error(f"Could not render graph: {e}")

        # 3. Generation (Streaming)
        full_res = ""
        hist_tuples = [(m["role"], m["content"]) for m in st.session_state.messages]
        
        try:
            for chunk in stream_answer(prompt, docs, hist_tuples, model_choice, complexity):
                full_res += chunk
                msg_box.markdown(full_res + "▌")
            
            msg_box.markdown(full_res)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_res = "I encountered an error generating the response."
    
    st.session_state.messages.append({"role": "assistant", "content": full_res})