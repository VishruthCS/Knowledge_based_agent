# 🕸️ AI Knowledge Base Agent (V2.1)

### 🌐 **Live Demo:**  
🚀 **https://knowledgebasedagent-hmsavacrqcpalcfygerxnv.streamlit.app/**  

A powerful **Retrieval-Augmented Generation (RAG)** application built with Python, Streamlit, and Google Gemini. This agent allows users to upload documents (PDF, DOCX, TXT), creates a temporary knowledge base, and answers questions with citation-backed evidence and knowledge graph visualizations.

---

## 🚀 Key Features

* **⚡ Session-Specific Memory:** Creates a unique, temporary vector store for every browser session. Data is isolated and automatically cleared when the session ends.
* **🧠 Multi-Model Support:** Choose between `gemini-2.5-flash`, `gemini-2.5-pro`, and `gemini-2.0-flash` for different reasoning capabilities.
* **📄 Reasoning Trace:** Provides exact source citations (filename, page number, and snippets) for every answer to ensure accuracy.
* **🕸️ Knowledge Graph Generation:** Automatically extracts entities and relationships from the retrieved context and visualizes them using Graphviz.
* **✍️ Intelligent Query Rewriting:** Uses an intermediate LLM step to refine vague user queries into search-optimized keywords.
* **🎚️ Adaptive Complexity:** Adjust answers based on the target audience (Novice, Standard, Expert).
* **📁 Robust File Parsing:** Supports OCR (Tesseract) for scanned PDFs, `pdfplumber` for text PDFs, and `python-docx` for Word documents.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit  
* **LLM Provider:** Google Gemini API  
* **Orchestration:** LangChain  
* **Vector Database:** ChromaDB  
* **Embeddings:** Google `embedding-001`  
* **Visualization:** Graphviz  
* **OCR/Parsing:** pdf2image, pytesseract, pdfplumber  

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.11
* Docker Desktop  
* A Google AI Studio API Key  

---

## ▶️ Option 1: Run with Docker (Recommended)

```bash
git clone https://github.com/VishruthCS/Knowledge_based_agent.git
cd Knowledge_based_agent
```

Create `.env` file:

```
GOOGLE_API_KEY="your_actual_api_key_here"
```

Run container:

```bash
docker-compose up --build
```

Open app:  
👉 `http://localhost:8501`

---

## ▶️ Option 2: Local Installation (Manual)

### **1. Install System Dependencies**

**Windows:**  
- Install Tesseract OCR  
- Install Poppler  
- Add both to PATH  

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get update && sudo apt-get install -y build-essential poppler-utils tesseract-ocr
```

**Mac (Homebrew):**

```bash
brew install tesseract poppler
```

---

### **2. Install Python Dependencies**

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **3. Run the Application**

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
├── app.py
├── backend.py
├── ingest.py
├── model.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
└── uploads/
```

---

## 🖥️ Usage Guide

1. Upload PDF, DOCX, TXT  
2. Click **Process & Index**  
3. Ask questions  
4. View **Reasoning Trace** and **Knowledge Graph**  
5. Adjust settings (complexity, model, rewriting)

---

## 🛡️ Troubleshooting

**“Microsoft Visual C++ 14.0 required”**  
→ Use Docker OR install Build Tools  

**“Tesseract not installed”**  
→ Install and add to PATH  

**“Gemini model not found”**  
→ Check API key access in Google AI Studio  

---


