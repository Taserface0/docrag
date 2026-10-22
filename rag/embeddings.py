"""Embedding backends. Ollama for local use; a hashing embedder for offline tests.

Embedders expose two methods:
  embed(texts)       -> vectors for document chunks (used at indexing time)
  embed_query(text)  -> vector for a search query (used at question time)
Some models (e.g. nomic-embed-text) are trained with different task prefixes
for documents and queries, so the two must not be embedded the same way.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from typing import Protocol


class Embedder(Protocol):
    fingerprint: str  # identifies the vector space; stored with the index

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


# Task prefixes recommended by the model authors: (document_prefix, query_prefix)
KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
}


def default_prefixes(model: str) -> tuple[str, str]:
    base = model.split(":")[0].split("/")[-1]
    return KNOWN_PREFIXES.get(base, ("", ""))


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        batch_size: int = 32,
        doc_prefix: str | None = None,
        query_prefix: str | None = None,
        timeout: int = 300,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        auto_doc, auto_query = default_prefixes(model)
        self.doc_prefix = auto_doc if doc_prefix is None else doc_prefix
        self.query_prefix = auto_query if query_prefix is None else query_prefix
        self._legacy = False  # set if the server only has the old /api/embeddings endpoint

    @property
    def fingerprint(self) -> str:
        return f"ollama:{self.model}|doc:{self.doc_prefix}"

    # ---------- public ----------
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed_many([self.doc_prefix + self._clean(t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed_many([self.query_prefix + self._clean(text)])[0]

    # ---------- internals ----------
    @staticmethod
    def _clean(text: str) -> str:
        return text.replace("\n", " ")

    def _post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            model_missing = "model" in detail.lower() and "not found" in detail.lower()
            if error.code == 404 and path == "/api/embed" and not model_missing:
                raise _EndpointMissing from error  # old server without /api/embed
            hint = f" Run `ollama pull {self.model}`." if model_missing else ""
            raise RuntimeError(f"Ollama embedding request failed ({error.code}): {detail}{hint}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}: {error.reason}. Is Ollama running?"
            ) from error

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            if not self._legacy:
                try:
                    result = self._post("/api/embed", {"model": self.model, "input": batch, "truncate": True})
                    out.extend(result["embeddings"])
                    continue
                except _EndpointMissing:
                    self._legacy = True  # older Ollama: fall back to one request per text
            for text in batch:
                result = self._post("/api/embeddings", {"model": self.model, "prompt": text})
                out.append(result["embedding"])
        return out


class _EndpointMissing(Exception):
    pass


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. No network; used in tests."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.fingerprint = f"hash:{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / n for x in v])
        return vecs

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
