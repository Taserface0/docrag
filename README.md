# DocRAG — Chat with your PDFs, with citations

A retrieval-augmented generation (RAG) system built from scratch in Python. Upload PDFs, ask questions, and get answers that cite the exact file and page they came from. Retrieval is **hybrid** (dense embeddings + BM25 keyword search, fused with Reciprocal Rank Fusion) with **cross-encoder reranking**, and the repo ships an **evaluation suite** that measures each retriever so design choices are backed by numbers.

No LangChain or LlamaIndex: every stage is a small, readable module.

## Architecture

```
            ┌──────────── INGEST ────────────┐
 PDF ──► pypdf text per page ──► recursive chunker (size + overlap, keeps page no.)
                                        │
                     ┌──────────────────┴──────────────────┐
              Ollama embeddings                       BM25 index
                     │                                     │
               Chroma (cosine)                     (rebuilt on change)
            └──────────────────────────────────────────────┘

            ┌──────────── QUERY ─────────────┐
 question ──► condense follow-up with chat history ──► standalone query
          ──► dense top-20  ┐
          ──► BM25 top-20   ┴► Reciprocal Rank Fusion ──► [cross-encoder rerank] ──► top-k
          ──► LLM with numbered sources ──► answer with [n] citations (streamed)
```

| Module | Responsibility |
|---|---|
| `rag/ingest.py` | PDF → cleaned page text → overlapping chunks with source/page metadata; content-hash doc IDs (re-uploads are no-ops) |
| `rag/embeddings.py` | Batched Ollama embeddings with model-specific document/query prefixes; a hashing embedder for offline tests |
| `rag/store.py` | Persistent Chroma collection + BM25 over the same chunks; per-document filtering and deletion |
| `rag/retriever.py` | Dense / BM25 / hybrid (RRF) retrieval, cross-encoder rerank |
| `rag/generator.py` | Grounded prompt, citation parsing, follow-up question condensing |
| `rag/pipeline.py` | Orchestration used by the UI, API and evals |
| `app.py` | Streamlit chat UI |
| `api.py` | FastAPI REST API |
| `eval/` | Synthetic test-set generation and retrieval/answer metrics |

## Why these design choices

- **Hybrid retrieval.** Embeddings catch paraphrases ("how do cells make energy" → mitochondria) but miss exact tokens such as error codes, IDs and names; BM25 is the reverse. RRF merges the two ranked lists using ranks only, so their incompatible score scales don't matter.
- **Reranking.** A cross-encoder reads the query and passage together, which is more accurate than comparing two independent embeddings, but too slow to run over the whole corpus. Running it only on the ~40 fused candidates gets most of the accuracy for little latency.
- **Question condensing.** "What about its side effects?" is a bad search query. The LLM rewrites follow-ups into standalone questions before retrieval.
- **Grounding and citations.** Sources are numbered in the prompt; the model must cite `[n]` and say when the answer isn't in the documents. Citations outside the provided range are discarded, and the UI shows only the passages actually cited.
- **Measured, not assumed.** `eval/` compares dense vs BM25 vs hybrid vs hybrid+rerank on hit@k and MRR.

## Quick start

### Option A: Docker (easiest; nothing else to install)

Needs only [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/Taserface0/docrag.git
cd docrag
docker compose up --build
```

Open http://localhost:8501. The first start downloads the Python libraries, Ollama and both models (~2.5 GB), so it takes a while; later starts take seconds. Models and your index are kept in Docker volumes between runs.

- REST API too: `docker compose --profile api up --build`, then http://localhost:8000/docs
- Different models: create a `.env` with `CHAT_MODEL=...` / `EMBED_MODEL=...`
- NVIDIA GPU: uncomment the `deploy:` block in `docker-compose.yml` for much faster answers
- Stop: `Ctrl+C`, or `docker compose down` (add `-v` to also delete models and the index)

### Option B: Run directly with Python

Install [Ollama](https://ollama.com) first.

```bash
python -m venv .venv
.venv\Scripts\activate                  # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
copy .env.example .env                  # macOS/Linux: cp .env.example .env

streamlit run app.py                    # UI on http://localhost:8501
uvicorn api:app --reload                # API on http://localhost:8000/docs
```

DocRAG uses local Ollama models for chat and embeddings; no API key is needed. Set `OLLAMA_BASE_URL`, `CHAT_MODEL` and `EMBED_MODEL` in `.env` to use a different Ollama server or models.

Reranking with a cross-encoder is on by default (`USE_RERANKER=true`). Its ~90 MB model downloads on first use. Turn it off in the UI toggle or with `USE_RERANKER=false`.

The index remembers which embedding model built it. If you change `EMBED_MODEL` (or the embedding prefixes), the app asks you to clear the index and re-index, because vectors from different models can't be compared.

## API

| Method | Path | Body |
|---|---|---|
| `POST` | `/documents` | multipart `file` (PDF) |
| `GET` | `/documents` | — |
| `DELETE` | `/documents/{doc_id}` | — |
| `POST` | `/ask` | `{"question": "...", "history": [], "doc_ids": null, "top_k": 5, "mode": "hybrid", "rerank": null}` |

```bash
curl -F file=@paper.pdf localhost:8000/documents
curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"What dataset did they use?"}'
```

## Evaluation

```bash
python -m eval.generate_testset --n 40          # LLM writes one question per sampled chunk
python -m eval.evaluate --k 5 --rerank --judge  # retrieval table (rerank: all vs top 20) + LLM-judged answers
```

Results on a 41-page project report (30 synthetic questions, k = 5, laptop CPU, `nomic-embed-text`):

```
retriever                 | hit@1 | hit@5 | MRR   | page@5 | ms/query
dense                     | 0.467 | 0.733 | 0.567 | 0.900  | 65
bm25                      | 0.533 | 0.767 | 0.631 | 0.800  | 0.7
hybrid                    | 0.533 | 0.800 | 0.631 | 0.900  | 60
hybrid + rerank (all)     | 0.533 | 0.900 | 0.696 | 0.967  | 1889
hybrid + rerank (top 20)  | 0.533 | 0.900 | 0.696 | 0.967  | 1266   <- default
```

- **Hybrid beats either method alone** (80% vs 73% dense, 77% BM25). BM25 does well here because the report is full of exact names (SmartRecco, ChromaDB, Django); combining both catches what each one misses.
- **Reranking gives the biggest gain:** the cross-encoder lifts top-5 accuracy from 80% to **90%** and MRR from 0.63 to 0.70, mostly by pulling relevant passages from outside the top 5 into it.
- **Reranking only the top 20 fused candidates keeps identical accuracy and is 33% faster** (1.9 s → 1.3 s per query on CPU), so it's the default (`RERANK_CANDIDATES=20`). Turn reranking off with `USE_RERANKER=false` when speed matters more.
- With 30 questions, one question is about 3 percentage points, so treat small gaps as indicative; a larger test set would make them more reliable.

Caveat: synthetic questions are written from a single chunk, so they slightly favour lexical overlap. Add a few hand-written questions to `eval/testset.jsonl` for a fairer picture.

## Tests

```bash
pytest -q
```

The tests run fully offline (hashing embedder + fake LLM) and cover chunking, RRF, idempotent ingest, BM25 exact-match lookups, document filtering/deletion, follow-up condensing, citation parsing and streaming.

## Limitations and next steps

- Scanned PDFs need OCR first (e.g. `ocrmypdf`); text-less PDFs are rejected with a clear message.
- On Windows, use `127.0.0.1` rather than `localhost` in `OLLAMA_BASE_URL`: `localhost` can try IPv6 first and add ~2 s to every Ollama call (measured: 2.1 s → 59 ms per query).
- BM25 is in memory and rebuilt after changes — fine for hundreds of documents; swap in a search engine (OpenSearch, Qdrant sparse vectors) beyond that.
- Ideas: semantic/heading-aware chunking, table extraction, query decomposition for multi-hop questions, answer caching, auth and per-user collections.

## Config

All settings are environment variables (see `.env.example`): models, chunk size/overlap, candidate count, top-k and reranker.
