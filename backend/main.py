import os
import shutil
import uuid
from dotenv import dotenv_values
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from backend.ingest import load_pdfs, split_documents, embed_and_store
from backend.rag_chain import build_rag_chain
from contextlib import asynccontextmanager
from backend.evaluate import evaluate_rag, log_to_mlflow
import time
import asyncio

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
    use_session: bool = False
    session_id: str = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RAG API is running"}

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    global chain, retriever
    
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
    chain, retriever = build_rag_chain()


    return {"message": f"Successfully ingested '{file.filename}'", "chunks": len(chunks)}


@app.post("/ingest-session")
async def ingest_session(files: list[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = f"sessions/{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")
        with open(f"{session_dir}/{file.filename}", "wb") as f:
            shutil.copyfileobj(file.file, f)

    docs = load_pdfs(session_dir)
    chunks = split_documents(docs)
    embed_and_store(chunks, persist_dir=f"chroma_sessions/{session_id}")

    return {"session_id": session_id, "chunks": len(chunks)}



@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    global chain, retriever

    start = time.time()  # ← start timer

    if request.use_session and request.session_id:
        session_chain, session_retriever = build_rag_chain(
            chroma_path=f"chroma_sessions/{request.session_id}"
        )
        active_chain = session_chain
        active_retriever = session_retriever
    else:
        active_chain = chain
        active_retriever = retriever

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = active_chain.invoke(request.question)
    
    # Check if retriever is a function or has an .invoke method
    if callable(active_retriever) and not hasattr(active_retriever, "invoke"):
        source_docs = active_retriever(request.question)
    else:
        source_docs = active_retriever.invoke(request.question)

    latency = time.time() - start  # ← end timer

    sources = [
        {
            "page": doc.metadata.get("page", "?"),
            "source": os.path.basename(doc.metadata.get("source", "?")),
            "content": doc.page_content  # Send the actual text snippet
        }
        for doc in source_docs
    ]

    # Extract raw text from retrieved chunks for RAGAs
    contexts = [doc.page_content for doc in source_docs]  

    # Use FastAPI's BackgroundTasks
    background_tasks.add_task(
        _evaluate_and_log, 
        request.question, 
        answer, 
        contexts, 
        latency, 
        request.session_id or "global"
    )

    return QueryResponse(answer=answer, sources=sources)  # ← returns immediately


def _evaluate_and_log(question, answer, contexts, latency, session_id):
    try:
        scores = evaluate_rag(question, answer, contexts)
        log_to_mlflow(question, answer, scores, latency, session_id)
    except Exception as e:
        print(f"[MLflow] Evaluation failed: {e}")

