# Knowledge Base Agent — RAG Demo

## Overview
This is a minimal, judge-ready Knowledge Base Agent that allows uploading documents (PDF/DOCX/TXT),
indexes them using OpenAI embeddings into Chroma, and answers questions via a Retrieval-Augmented-Generation (RAG) pipeline.

## How to run (local)
1. Create virtual env & install:
   python -m venv venv
   source venv/bin/activate  # windows: venv\\Scripts\\activate
   pip install -r requirements.txt

2. Set environment variable:
   export OPENAI_API_KEY="sk-xxxx"
   (Or set in Streamlit secrets)

3. Run the app:
   streamlit run app.py

4. Workflow:
   - Upload a few sample PDFs/DOCX/TXT.
   - Click "Index / Re-Index uploaded files".
   - Ask questions in the query box and view answers + sources.

## Deliverables (for submission)
- Demo link: (hosted Streamlit app or local run instructions)
- Git repo: include app.py, ingest.py, requirements.txt, README, architecture diagram.
- Architecture diagram: show flow: Streamlit -> Ingest -> Embeddings -> Chroma -> LLM -> UI
- Short demo script (2 minutes) included below.

## Demo script (2-min):
1. Show app UI and uploaded files area (10s)
2. Upload 2-3 sample policy PDFs (10s)
3. Click "Index" — show progress (10s)
4. Ask 2 quick questions (30s): one factual question that’s directly supported; one requiring cross-file context.
5. Show sources with page and chunk snippet (20s)
6. Explain design choices: RAG, metadata, cost optimisation, re-index, compliance (40s)

## Notes & Extensions
- Add OCR (Tesseract) for scanned PDFs.
- Add authentication and per-user persistent storage for privacy.
- Add confidence scoring and an option to return exact quoted text segments.

