import uvicorn
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.search_engine import SearchEngine
from config import MODEL_NAME

app = FastAPI(title="Code Search API")
engine = None

class Document(BaseModel):
    text: str
    doc_id: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    top_k: int = 10

class SearchResult(BaseModel):
    doc_id: str
    score: float
    text: str

def _check_engine():
    if engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized. Call /initialize first")

@app.post("/initialize")
def initialize(model_name: str = MODEL_NAME):
    global engine
    engine = SearchEngine(model_name=model_name, collection_name="api_collection")
    return {"status": "initialized", "model": model_name}

@app.post("/add_documents")
def add_documents(documents: List[Document]):
    _check_engine()
    texts = [doc.text for doc in documents]
    doc_ids = [doc.doc_id if doc.doc_id else str(i) for i, doc in enumerate(documents)]
    engine.add_documents(texts, doc_ids=doc_ids)
    return {"status": "success", "count": len(documents)}

@app.post("/search", response_model=List[SearchResult])
def search(query: SearchQuery):
    _check_engine()
    return engine.search(query.query, top_k=query.top_k)

@app.get("/health")
def health():
    return {"status": "ok", "engine_initialized": engine is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

