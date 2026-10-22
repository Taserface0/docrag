import json

from fastapi.testclient import TestClient


def test_api_roundtrip(pipeline, sample_pdf, monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "apidata"))
    import api

    monkeypatch.setattr(api, "rag", pipeline)
    c = TestClient(api.app)

    r = c.post("/documents", files={"file": ("bio.pdf", sample_pdf, "application/pdf")})
    assert r.status_code == 200 and r.json()["chunks"] > 0
    assert c.post("/documents", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400

    docs = c.get("/documents").json()
    assert docs[0]["source"] == "bio.pdf"

    r = c.post("/ask", json={"question": "What is E-4012?", "mode": "hybrid"})
    body = r.json()
    assert r.status_code == 200 and body["sources"][0]["cited"] is True

    assert c.delete(f"/documents/{docs[0]['doc_id']}").status_code == 200
    assert c.delete("/documents/nope").status_code == 404


def test_eval_metrics(pipeline, sample_pdf, tmp_path):
    from eval.evaluate import retrieval_metrics

    pipeline.ingest_pdf(sample_pdf, "bio.pdf")
    chunks = pipeline.store.all_chunks()
    gold = next(c for c in chunks if "E-4012" in c.text)
    items = [{"question": "payment gateway error E-4012", "chunk_id": gold.id, "doc_id": gold.doc_id, "page": gold.page}]
    m = retrieval_metrics(pipeline, items, k=3, mode="hybrid", rerank=False)
    assert m["hit@1"] == 1.0 and m["MRR"] == 1.0
    json.dumps(m)
