# Paper RAG Assistant
A production-ready RAG system for medical research papers using LangChain, ChromaDB, and DeepSeek via OpenRouter.

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
