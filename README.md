# 🧬 Paper RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions about research papers with high accuracy. Built with LangChain, ChromaDB, Gemini, and monitored with RAGAs + MLflow.

## 🚀 Features

- **Multi-PDF support** — Upload one or more research papers per session
- **Accurate retrieval** — HuggingFace embeddings + ChromaDB vector store
- **Gemini-powered answers** — Fast, context-aware responses via Gemini API
- **RAG evaluation** — Automatic faithfulness, answer relevancy & context precision scoring via RAGAs
- **MLflow tracking** — Every query logged with metrics and latency
- **Session isolation** — Each user upload gets its own vector store

## 🛠️ Stack

| Layer | Technology |
|-------|-----------|
| LLM | Gemini API |
| Embeddings | HuggingFace |
| Vector Store | ChromaDB |
| RAG Framework | LangChain (LCEL) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Evaluation | RAGAs |
| Experiment Tracking | MLflow |

## ⚙️ Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/Thanos2002/paper-rag-assistant.git
cd medical-rag-assistant
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Add your keys to `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openrouter_key   # used for RAGAs evaluation
```

### 3. Ingest your PDFs

```bash
python backend/ingest.py
```

### 4. Run the backend

```bash
uvicorn backend.main:app --reload
```

### 5. Run the frontend

```bash
streamlit run frontend/app.py
```

### 6. Launch MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000` to monitor evaluation metrics.

## 📁 Project Structure

```
medical-rag-assistant/
├── backend/
│   ├── main.py          # FastAPI endpoints
│   ├── rag_chain.py     # LangChain RAG pipeline
│   ├── ingest.py        # PDF ingestion & chunking
│   └── evaluate.py      # RAGAs evaluation + MLflow logging
├── frontend/
│   └── app.py           # Streamlit UI
├── chroma_db/           # Persistent vector store (auto-generated)
├── mlflow.db            # MLflow experiment tracking (auto-generated)
├── .env.example
├── requirements.txt
└── README.md
```

## 📊 Evaluation Metrics

Each query is automatically evaluated and logged to MLflow:

- **Faithfulness** — Are the answers grounded in the retrieved context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — Were the retrieved chunks relevant?

## 🔑 API Keys

| Key | Where to get it |
|-----|----------------|
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |
| `OPENAI_API_KEY` | [openrouter.ai](https://openrouter.ai) (free tier) |
