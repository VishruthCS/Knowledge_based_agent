import os
import traceback
import concurrent.futures
from typing import List, Dict, Any, Generator

# File Processing
from PIL import Image
import pytesseract

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import docx
except ImportError:
    docx = None

# LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Sentence Transformers Fallback
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# ==========================================
# 1. TEXT EXTRACTION (Parallelizable)
# ==========================================

def extract_text_single_file(file_path: str) -> List[Dict]:
    """Extracts text from a single file. Used by the thread pool."""
    pages = []
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf" and pdfplumber:
            with pdfplumber.open(file_path) as pdf:
                for i, p in enumerate(pdf.pages):
                    text = p.extract_text() or ""
                    if text.strip():
                        pages.append({"page": i + 1, "text": text})
        
        elif ext in [".docx", ".doc"] and docx:
            doc = docx.Document(file_path)
            full_text = "\n".join([p.text for p in doc.paragraphs])
            if full_text.strip():
                pages.append({"page": 1, "text": full_text})
        
        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                pages.append({"page": 1, "text": f.read()})
                
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return pages

# ==========================================
# 2. EMBEDDING STRATEGY
# ==========================================

class LocalSentenceEmbedding:
    """Fallback if Gemini API is down or Key is missing."""
    def __init__(self, model="all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed.")
        self.m = SentenceTransformer(model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self.m.encode(texts, convert_to_numpy=True)]

    def embed_query(self, text: str) -> List[float]:
        return self.m.encode([text], convert_to_numpy=True)[0].tolist()

def get_embeddings_function():
    """Returns the embedding function (Gemini or Local)."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            # Check debug output: 'models/embedding-001' IS available in your list
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
        except Exception:
            pass
    
    print("⚠️ Using Local Embeddings (Fallback)")
    return LocalSentenceEmbedding()

# ==========================================
# 3. CORE LOGIC (Ingest & Query)
# ==========================================

def process_and_index(file_paths: List[str], persist_directory="chroma_db"):
    """
    Reads files in PARALLEL to speed up ingestion.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_docs = []

    # Parallel processing using ThreadPool
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(extract_text_single_file, fp): fp for fp in file_paths}
        
        for future in concurrent.futures.as_completed(future_to_file):
            fp = future_to_file[future]
            try:
                pages = future.result()
                for p in pages:
                    chunks = text_splitter.split_text(p['text'])
                    for i, chunk in enumerate(chunks):
                        all_docs.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": os.path.basename(fp),
                                "file_path": fp,
                                "page": p['page'],
                                "chunk_id": i
                            }
                        ))
            except Exception as e:
                print(f"Failed to process {fp}: {e}")

    if not all_docs:
        return {"success": False, "message": "No text extracted from files."}

    # Indexing
    emb_fn = get_embeddings_function()
    try:
        # Batch add documents to Chroma
        vectordb = Chroma.from_documents(
            documents=all_docs,
            embedding=emb_fn,
            persist_directory=persist_directory
        )
        vectordb.persist()
        return {"success": True, "message": f"Successfully indexed {len(all_docs)} chunks."}
    except Exception as e:
        return {"success": False, "message": str(e)}

def get_llm(model_name="gemini-2.5-flash"):
    """
    Creates LLM with fallback logic specific to your available models.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Try the requested model first
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        return llm
    except Exception:
        pass
        
    # If instantiation fails immediately, return a safer default from your list
    print(f"Warning: Could not load {model_name}, falling back to gemini-2.0-flash")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True
    )

def query_stream(question: str, chat_history: List, persist_directory="chroma_db", model_name="gemini-2.5-flash") -> Generator:
    """
    Yields chunks of text for the streaming effect.
    """
    emb_fn = get_embeddings_function()
    
    # Load Vector DB
    try:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=emb_fn)
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        yield f"Error loading index: {e}"
        return

    # 1. Retrieve Context
    try:
        docs = retriever.invoke(question)
    except Exception:
        docs = retriever.get_relevant_documents(question)

    context_text = "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}" for d in docs])
    
    if not context_text:
        yield "I couldn't find any relevant information in the uploaded documents."
        return

    # 2. Format Chat History
    history_text = ""
    for role, text in chat_history[-4:]: 
        history_text += f"{role}: {text}\n"

    # 3. Construct Prompt
    system_prompt = (
        "You are a helpful Knowledge Base Assistant. "
        "Use the following Context to answer the user's question. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"CHAT HISTORY:\n{history_text}\n\n"
        f"USER QUESTION: {question}\n\n"
        "ANSWER:"
    )

    # 4. Stream Response with Fallback
    llm = get_llm(model_name)
    
    try:
        try:
            for chunk in llm.stream(system_prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            # Check for 404 (Model Not Found)
            err_msg = str(e)
            if "404" in err_msg or "NotFound" in err_msg:
                # Fallback Level 1: gemini-2.0-flash (Available in your list)
                yield f"⚠️ Model '{model_name}' not found. Trying 'gemini-2.0-flash'...\n\n"
                
                try:
                    fallback_llm = get_llm("gemini-2.0-flash")
                    for chunk in fallback_llm.stream(system_prompt):
                        if chunk.content:
                            yield chunk.content
                except Exception as e2:
                    if "404" in str(e2):
                         # Fallback Level 2: gemini-flash-latest (Generic alias)
                        yield "⚠️ 'gemini-2.0-flash' also failed. Trying 'gemini-flash-latest'...\n\n"
                        fallback_llm_2 = get_llm("gemini-flash-latest")
                        for chunk in fallback_llm_2.stream(system_prompt):
                            if chunk.content:
                                yield chunk.content
                    else:
                        raise e2
            else:
                raise e
                
    except Exception as final_error:
        yield f"❌ All models failed. Error: {final_error}\nPlease check your API Key and Google Cloud Project settings."