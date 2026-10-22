"""REST API over the same pipeline.

Run:  uvicorn api:app --reload      Docs at http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag import RAGPipeline

app = FastAPI(title="DocRAG API", version="1.0.0")
rag = RAGPipeline()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[Message] = []
    doc_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    rerank: bool | None = None


class Source(BaseModel):
    n: int
    source: str
    page: int
    text: str
    score: float
    cited: bool


class AskResponse(BaseModel):
    answer: str
    search_query: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": rag.store.count()}


@app.post("/documents")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    try:
        r = rag.ingest_pdf(await file.read(), file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return r.__dict__


@app.get("/documents")
def list_documents():
    return rag.list_docs()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not rag.store.has_doc(doc_id):
        raise HTTPException(404, "Document not found")
    rag.delete_doc(doc_id)
    return {"deleted": doc_id}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    a = rag.ask(
        req.question,
        history=[m.model_dump() for m in req.history],
        doc_ids=req.doc_ids,
        top_k=req.top_k,
        mode=req.mode,
        rerank=req.rerank,
    )
    return AskResponse(
        answer=a.text,
        search_query=a.search_query,
        sources=[
            Source(n=i, source=h.chunk.source, page=h.chunk.page, text=h.chunk.text, score=h.score, cited=i in a.cited)
            for i, h in enumerate(a.sources, start=1)
        ],
    )
