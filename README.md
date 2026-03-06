# Paper RAG Assistant
A production-ready RAG system for that ansewrs your questions about your research papers accurately using LangChain, ChromaDB, and Gemini.

## Stack
- LangChain (LCEL) + ChromaDB + HuggingFace Embeddings
- Gemini API
- FastAPI + Streamlit + MLflow

## Setup
```
pip install -r requirements.txt
cp .env.example .env  # add your OpenRouter key
python backend/ingest.py
python backend/rag_chain.py
```
