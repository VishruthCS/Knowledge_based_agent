import os
import streamlit as st
from backend import process_and_index, query_stream

# Constants
UPLOAD_DIR = "uploads"
PERSIST_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Page Config
st.set_page_config(page_title="AI Knowledge Agent", layout="wide")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ==========================================
# SIDEBAR: CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("🧠 Agent Brain")
    
    # Model Selector
    # Updated based on your 'debug.py' output
    model_choice = st.selectbox(
        "Choose Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        index=0,
        help="Select a model available to your API key."
    )
    
    st.divider()
    
    st.header("📂 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, Docx, Txt)", 
        accept_multiple_files=True
    )
    
    if st.button("⚡ Process & Index Documents"):
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            with st.spinner("Processing files in parallel..."):
                # Save files to disk
                saved_paths = []
                for f in uploaded_files:
                    path = os.path.join(UPLOAD_DIR, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    saved_paths.append(path)
                
                # Call Backend
                result = process_and_index(saved_paths, PERSIST_DIR)
                
                if result["success"]:
                    st.success(result["message"])
                    st.session_state.processed_files = saved_paths
                    st.session_state.messages = [] # Clear chat on new index
                else:
                    st.error(result["message"])

    if st.session_state.processed_files:
        st.write(f"📚 Indexed {len(st.session_state.processed_files)} files")
    
    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================
st.title("🤖 Intelligent Knowledge Agent")
st.caption("Powered by Gemini • Vector Search • RAG")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        history_for_backend = [(m["role"], m["content"]) for m in st.session_state.messages]
        
        try:
            stream_gen = query_stream(
                prompt, 
                history_for_backend, 
                PERSIST_DIR, 
                model_name=model_choice
            )
            
            for chunk in stream_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "I encountered an error processing your request."

    # Add assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})