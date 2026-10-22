"""High-level RAG pipeline used by the Streamlit app, the API and the eval scripts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .config import Settings
from .embeddings import Embedder, OllamaEmbedder
from .generator import LLM, OllamaChat, build_messages, cited_indices, condense_question, needs_rewrite
from .ingest import chunk_pages, doc_id_for, load_pdf
from .retriever import Retriever
from .store import Hit, Store


@dataclass
class IngestResult:
    doc_id: str
    source: str
    pages: int
    chunks: int
    skipped: bool = False  # already indexed


@dataclass
class Answer:
    question: str
    search_query: str
    text: str
    sources: list[Hit]
    cited: list[int] = field(default_factory=list)  # 1-based indices into sources


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ):
        self.s = settings or Settings()
        self.embedder = embedder or OllamaEmbedder(
            self.s.embed_model,
            self.s.ollama_base_url,
            batch_size=self.s.embed_batch_size,
            doc_prefix=self.s.embed_doc_prefix,
            query_prefix=self.s.embed_query_prefix,
        )
        self.llm = llm or OllamaChat(self.s.chat_model, self.s.ollama_base_url)
        self.store = Store(self.s.data_dir, self.s.collection, embed_fingerprint=self.embedder.fingerprint)
        self.retriever = Retriever(
            self.store, self.embedder, candidates=self.s.candidates,
            reranker_model=self.s.reranker_model,
            rerank_candidates=self.s.rerank_candidates,
        )

    # ---------- documents ----------
    def ingest_pdf(self, data: bytes, filename: str) -> IngestResult:
        did = doc_id_for(data)
        if self.store.has_doc(did):
            return IngestResult(did, filename, 0, 0, skipped=True)
        pages = load_pdf(data, filename)
        if not pages:
            raise ValueError(
                f"No extractable text in {filename}. It may be a scanned PDF; run OCR on it first."
            )
        chunks = chunk_pages(pages, self.s.chunk_size, self.s.chunk_overlap)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        return IngestResult(did, filename, len(pages), len(chunks))

    def list_docs(self) -> list[dict]:
        return self.store.list_docs()

    def delete_doc(self, doc_id: str) -> None:
        self.store.delete_doc(doc_id)

    # ---------- Q&A: individual steps (the UI calls these one by one to show progress) ----------
    @staticmethod
    def needs_rewrite(question: str, history: list[dict] | None = None) -> bool:
        return needs_rewrite(question, history)

    def rewrite_query(self, question: str, history: list[dict] | None = None) -> str:
        """Turn a follow-up into a standalone search query. No-op for the first question."""
        return condense_question(self.llm, question, history)

    def search(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        mode: str = "hybrid",
        rerank: bool | None = None,
    ) -> list[Hit]:
        return self.retriever.retrieve(
            query,
            top_k=top_k or self.s.top_k,
            doc_ids=doc_ids,
            mode=mode,
            rerank=self.s.use_reranker if rerank is None else rerank,
        )

    def generate_stream(self, question: str, hits: list[Hit], history: list[dict] | None = None) -> Iterator[str]:
        if not hits:
            return iter(["I couldn't find that in the uploaded documents."])
        return self.llm.stream(build_messages(question, hits, history))

    # ---------- Q&A: all steps at once ----------
    def _retrieve(self, question, history, doc_ids, top_k, mode, rerank):
        query = self.rewrite_query(question, history)
        return query, self.search(query, doc_ids, top_k, mode, rerank)

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        mode: str = "hybrid",
        rerank: bool | None = None,
    ) -> Answer:
        query, hits = self._retrieve(question, history, doc_ids, top_k, mode, rerank)
        if not hits:
            return Answer(question, query, "I couldn't find that in the uploaded documents.", [])
        # Answer the standalone version: a small model can't reliably resolve "that"/"it" itself.
        text = self.llm.complete(build_messages(query, hits, history))
        return Answer(question, query, text, hits, cited_indices(text, len(hits)))

    def ask_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        mode: str = "hybrid",
        rerank: bool | None = None,
    ) -> tuple[str, list[Hit], Iterator[str]]:
        """Returns (search_query, sources, token_stream). Sources are known before generation starts."""
        query, hits = self._retrieve(question, history, doc_ids, top_k, mode, rerank)
        return query, hits, self.generate_stream(query, hits, history)
