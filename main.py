import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import ingest_pdf, get_answer
from dotenv import load_dotenv

# 1. Load the secret API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Fail fast if the API key is missing on startup
if not GROQ_API_KEY:
    raise ValueError("FATAL ERROR: GROQ_API_KEY is missing from environment variables!")

# 2. Create the server app
app = FastAPI(title="EduRAG API")

# 3. Security setting: Allow our future UI to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Create a folder to hold uploaded PDFs temporarily
os.makedirs("pdfs", exist_ok=True)

class QueryRequest(BaseModel):
    query: str

# --- ENDPOINT 1: Uploading a PDF ---
@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """Catches the uploaded PDF and sends it to our RAG Engine."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Generate a unique filename to prevent overwrite collisions
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join("pdfs", unique_filename)
    
    # Save the file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
    
    try:
        # Send it to the engine to be chopped and vectorized
        num_chunks = ingest_pdf(file_path)
        return {"message": f"Success! {file.filename} was chopped into {num_chunks} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the raw file after it's saved in the database
        if os.path.exists(file_path):
            os.remove(file_path)

# --- ENDPOINT 2: Asking a Question ---
@app.post("/ask")
def ask_question(request: QueryRequest):
    """Catches the question and gets the answer from our RAG Engine."""
    try:
        answer = get_answer(request.query, GROQ_API_KEY)
        return {"query": request.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
