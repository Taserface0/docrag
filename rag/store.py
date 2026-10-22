"""Persistent vector store (Chroma) plus an in-memory BM25 index over the same chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from .ingest import Chunk

_STOP = set(
    "a an the and or of to in on for is are was were be been by with as at from that this it its "
    "into than then there these those which who whom what when where why how do does did not no".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float


def _chunk_from(cid: str, doc: str, meta: dict) -> Chunk:
    return Chunk(cid, meta["doc_id"], meta["source"], int(meta["page"]), int(meta["index"]), doc)


class EmbeddingMismatchError(RuntimeError):
    """The index on disk was built with a different embedding model than the one configured."""


class Store:
    def __init__(self, path: Path, collection: str = "docrag", embed_fingerprint: str | None = None):
        self.client = chromadb.PersistentClient(path=str(path / "chroma"))
        meta = {"hnsw:space": "cosine"}
        if embed_fingerprint:
            meta["embed_model"] = embed_fingerprint
        self.col = self.client.get_or_create_collection(collection, metadata=meta)
        if embed_fingerprint:
            self._check_fingerprint(collection, embed_fingerprint, meta, path)
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[Chunk] = []

    def _check_fingerprint(self, name: str, fingerprint: str, meta: dict, path: Path) -> None:
        stored = (self.col.metadata or {}).get("embed_model")
        if stored == fingerprint:
            return
        if self.col.count() == 0:
            # Empty collection from an older version (or a different model): safe to recreate.
            self.client.delete_collection(name)
            self.col = self.client.create_collection(name, metadata=meta)
            return
        built_with = stored or "an unrecorded embedding model (older DocRAG version)"
        raise EmbeddingMismatchError(
            f"The index in {path} was built with {built_with}, but the current setting is "
            f"{fingerprint}. Vectors from different models can't be mixed. Either delete the "
            f"'{path / 'chroma'}' folder and re-index your PDFs, or set COLLECTION to a new name."
        )

    # ---------- writes ----------
    def has_doc(self, doc_id: str) -> bool:
        return bool(self.col.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        for i in range(0, len(chunks), 500):
            batch = chunks[i : i + 500]
            self.col.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
                embeddings=embeddings[i : i + 500],
            )
        self._bm25 = None

    def delete_doc(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})
        self._bm25 = None

    # ---------- reads ----------
    def list_docs(self) -> list[dict]:
        res = self.col.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for m in res["metadatas"]:
            d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "source": m["source"], "chunks": 0, "pages": 0})
            d["chunks"] += 1
            d["pages"] = max(d["pages"], int(m["page"]))
        return sorted(docs.values(), key=lambda d: d["source"].lower())

    def all_chunks(self) -> list[Chunk]:
        res = self.col.get(include=["documents", "metadatas"])
        chunks = [_chunk_from(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: (c.doc_id, c.index))

    def count(self) -> int:
        return self.col.count()

    @staticmethod
    def _where(doc_ids: list[str] | None):
        if not doc_ids:
            return None
        return {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}

    def dense_search(self, query_vec: list[float], k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        n = self.count()
        if n == 0:
            return []
        res = self.col.query(
            query_embeddings=[query_vec],
            n_results=min(k, n),
            where=self._where(doc_ids),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(_chunk_from(i, d, m), 1.0 - dist)  # cosine distance -> similarity
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def bm25_search(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        if self._bm25 is None:
            self._bm25_chunks = self.all_chunks()
            if not self._bm25_chunks:
                return []
            self._bm25 = BM25Okapi([tokenize(c.text) for c in self._bm25_chunks])
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        allowed = set(doc_ids) if doc_ids else None
        ranked = sorted(
            (
                (s, c)
                for s, c in zip(scores, self._bm25_chunks)
                if s > 0 and (allowed is None or c.doc_id in allowed)
            ),
            key=lambda x: -x[0],
        )
        return [Hit(c, float(s)) for s, c in ranked[:k]]
