from pathlib import Path
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import UPLOAD_DIR
from .document_parser import extract_text, chunk_text
from .knowledge_base import kb
from .llm import generate_answer
from .schemas import ChatRequest, ChatResponse

app = FastAPI(title="MindVault API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED = {".pdf", ".docx", ".txt"}

@app.get("/")
def root():
    return {"message": "MindVault API is running", "docs": "/docs"}

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")
    safe_name = Path(file.filename).name
    destination = UPLOAD_DIR / safe_name
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text = extract_text(destination)
        chunks = chunk_text(text)
        if not chunks:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="No readable text found in this file.")
        count = kb.add_document(safe_name, chunks)
        return {"message": "Document indexed successfully", "filename": safe_name, "chunks": count}
    except HTTPException:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not process document: {exc}")

@app.get("/api/documents")
def list_documents():
    return kb.list_documents()

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    sources = kb.search(request.question, top_k=5)
    answer = generate_answer(request.question, sources)
    public_sources = [
        {"filename": item["filename"], "snippet": item["text"][:320], "score": round(item["score"], 3)}
        for item in sources
    ]
    return {"answer": answer, "sources": public_sources}

@app.post("/api/summarize/{filename}")
def summarize(filename: str):
    matches = [item for item in kb.chunks if item["filename"] == filename]
    if not matches:
        raise HTTPException(status_code=404, detail="Document not found.")
    # Retrieve the first few chunks as a compact summary context.
    selected = matches[:8]
    prompt_question = "Summarize the key ideas, important facts, and takeaways from this document."
    answer = generate_answer(prompt_question, selected)
    return {"filename": filename, "summary": answer}
