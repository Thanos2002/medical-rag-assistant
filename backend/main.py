import os
import shutil
from dotenv import dotenv_values
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from backend.ingest import load_pdfs, split_documents, embed_and_store
from backend.rag_chain import build_rag_chain
from contextlib import asynccontextmanager

config = dotenv_values(".env")
chain = None
retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup
    global chain, retriever
    chain, retriever = build_rag_chain()
    yield
    # Runs once at shutdown (cleanup goes here)

app = FastAPI(lifespan=lifespan)


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RAG API is running"}

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file to data/
    file_path = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Ingest into ChromaDB
    docs = load_pdfs("data")
    chunks = split_documents(docs)
    embed_and_store(chunks)

    return {"message": f"Successfully ingested '{file.filename}'", "chunks": len(chunks)}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    global chain, retriever

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = chain.invoke(request.question)

    source_docs = retriever.invoke(request.question)
    sources = [
        {
            "page": doc.metadata.get("page", "?"),
            "source": os.path.basename(doc.metadata.get("source", "?"))
        }
        for doc in source_docs
    ]

    return QueryResponse(answer=answer, sources=sources)

