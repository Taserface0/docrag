"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion, optionally reranked."""
from __future__ import annotations

from functools import lru_cache

from .embeddings import Embedder
from .store import Hit, Store


def reciprocal_rank_fusion(result_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """RRF: score(d) = sum over lists of 1 / (k + rank). Rank-based, so it needs no score calibration."""
    scores: dict[str, float] = {}
    chunks = {}
    for hits in result_lists:
        for rank, h in enumerate(hits, start=1):
            scores[h.chunk.id] = scores.get(h.chunk.id, 0.0) + 1.0 / (k + rank)
            chunks[h.chunk.id] = h.chunk
    return [Hit(chunks[cid], s) for cid, s in sorted(scores.items(), key=lambda x: -x[1])]


@lru_cache(maxsize=2)
def _load_cross_encoder(name: str):
    from sentence_transformers import CrossEncoder  # optional dependency

    return CrossEncoder(name)


class Retriever:
    MODES = ("hybrid", "dense", "bm25")

    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        candidates: int = 20,
        reranker_model: str | None = None,
        rerank_candidates: int = 20,
    ):
        self.store = store
        self.embedder = embedder
        self.candidates = candidates
        self.reranker_model = reranker_model
        # How many of the fused results the cross-encoder re-scores (0 = all of them).
        # The reranker is the slow step, so scoring fewer candidates is faster.
        self.rerank_candidates = rerank_candidates

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
        mode: str = "hybrid",
        rerank: bool = False,
        rerank_candidates: int | None = None,
    ) -> list[Hit]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        n = self.candidates
        if mode == "dense":
            hits = self.store.dense_search(self.embedder.embed_query(query), n, doc_ids)
        elif mode == "bm25":
            hits = self.store.bm25_search(query, n, doc_ids)
        else:
            dense = self.store.dense_search(self.embedder.embed_query(query), n, doc_ids)
            sparse = self.store.bm25_search(query, n, doc_ids)
            hits = reciprocal_rank_fusion([dense, sparse])

        if rerank and hits and self.reranker_model:
            n_rerank = self.rerank_candidates if rerank_candidates is None else rerank_candidates
            hits = self._rerank(query, hits[:n_rerank] if n_rerank > 0 else hits)
        return hits[:top_k]

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        model = _load_cross_encoder(self.reranker_model)
        scores = model.predict([(query, h.chunk.text) for h in hits])
        return [Hit(h.chunk, float(s)) for s, h in sorted(zip(scores, hits), key=lambda x: -x[0])]
