import json
import urllib.error
from io import BytesIO

import pytest

from rag.embeddings import HashEmbedder, OllamaEmbedder
from rag.generator import OllamaChat
from rag.store import EmbeddingMismatchError, Store


def _http_error(url, code, body):
    return urllib.error.HTTPError(url, code, "err", {}, BytesIO(body.encode()))


def test_embedder_batches_with_api_embed_and_prefixes(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        requests.append((request.full_url, body))
        return BytesIO(json.dumps({"embeddings": [[0.1, 0.2]] * len(body["input"])}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    emb = OllamaEmbedder("nomic-embed-text:latest", "http://localhost:11434/", batch_size=2)

    vectors = emb.embed(["first\ntext", "second", "third"])
    assert vectors == [[0.1, 0.2]] * 3
    # 3 texts, batch size 2 -> 2 requests, not 3
    assert [url for url, _ in requests] == ["http://localhost:11434/api/embed"] * 2
    assert requests[0][1]["input"] == ["search_document: first text", "search_document: second"]

    emb.embed_query("what is it?")
    assert requests[-1][1]["input"] == ["search_query: what is it?"]


def test_unknown_model_gets_no_prefix_and_prefix_can_be_overridden():
    assert OllamaEmbedder("all-minilm").doc_prefix == ""
    e = OllamaEmbedder("nomic-embed-text", doc_prefix="", query_prefix="Q: ")
    assert (e.doc_prefix, e.query_prefix) == ("", "Q: ")


def test_falls_back_to_legacy_endpoint_on_old_ollama(monkeypatch):
    urls = []

    def fake_urlopen(request, timeout):
        urls.append(request.full_url)
        if request.full_url.endswith("/api/embed"):
            raise _http_error(request.full_url, 404, "404 page not found")
        return BytesIO(json.dumps({"embedding": [1.0]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    emb = OllamaEmbedder("nomic-embed-text")
    assert emb.embed(["a", "b"]) == [[1.0], [1.0]]
    emb.embed(["c"])  # remembers the server is old; doesn't retry /api/embed
    assert urls.count("http://127.0.0.1:11434/api/embed") == 1


def test_missing_model_gives_pull_hint(monkeypatch):
    def fake_urlopen(request, timeout):
        raise _http_error(request.full_url, 404, '{"error":"model \\"nomic-embed-text\\" not found, try pulling it first"}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="ollama pull nomic-embed-text"):
        OllamaEmbedder("nomic-embed-text").embed(["x"])


def test_index_refuses_a_different_embedding_model(tmp_path, pipeline, sample_pdf):
    from rag import RAGPipeline

    pipeline.ingest_pdf(sample_pdf, "bio.pdf")  # built with HashEmbedder(256)
    with pytest.raises(EmbeddingMismatchError, match="hash:256"):
        RAGPipeline(pipeline.s, embedder=HashEmbedder(dim=128), llm=pipeline.llm)
    # same model reopens fine
    assert RAGPipeline(pipeline.s, embedder=HashEmbedder(), llm=pipeline.llm).store.count() > 0


def test_legacy_unlabelled_index(tmp_path):
    from rag.ingest import Chunk

    legacy = Store(tmp_path, "legacy")  # older version: no fingerprint recorded
    legacy.add([Chunk("x-0", "x", "a.pdf", 1, 0, "hello world text")], [[1.0, 0.0]])
    with pytest.raises(EmbeddingMismatchError, match="unrecorded"):
        Store(tmp_path, "legacy", embed_fingerprint="ollama:nomic")
    # an empty unlabelled collection is just recreated with the label
    Store(tmp_path, "empty")
    s = Store(tmp_path, "empty", embed_fingerprint="ollama:nomic")
    assert s.col.metadata["embed_model"] == "ollama:nomic"


def test_ollama_chat_complete_and_stream(monkeypatch):
    requests = []
    responses = [
        BytesIO(json.dumps({"message": {"content": "Complete answer"}}).encode()),
        BytesIO(
            b'{"message":{"content":"Stream "},"done":false}\n'
            b'{"message":{"content":"answer"},"done":true}\n'
        ),
    ]

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return responses.pop(0)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    chat = OllamaChat("qwen2.5:3b")
    messages = [{"role": "user", "content": "Hello"}]

    assert chat.complete(messages) == "Complete answer"
    assert "".join(chat.stream(messages)) == "Stream answer"
    assert requests[0]["stream"] is False
    assert requests[1]["stream"] is True
    assert requests[0]["model"] == "qwen2.5:3b"
