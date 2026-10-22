"""Build a synthetic Q&A test set from the indexed chunks.

For each sampled chunk the LLM writes one question answerable from that chunk alone,
plus a short reference answer. The chunk id becomes the retrieval ground truth.

    python -m eval.generate_testset --n 40 --out eval/testset.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

from rag import RAGPipeline

PROMPT = """You are writing an evaluation set for a document search system.
Read the passage and write ONE question that:
- can be answered from this passage alone,
- is specific (would not match most other passages),
- is phrased the way a real user would ask, without copying long phrases from the passage.

Return JSON: {{"question": "...", "answer": "<short reference answer>"}}
If the passage has no factual content (e.g. a table of contents), return {{"question": null}}.

Passage (from {source}, page {page}):
\"\"\"{text}\"\"\""""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="eval/testset.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rag = RAGPipeline()
    chunks = [c for c in rag.store.all_chunks() if len(c.text) > 200]
    if not chunks:
        raise SystemExit("Index is empty. Upload PDFs first.")
    random.Random(args.seed).shuffle(chunks)

    written = 0
    with open(args.out, "w") as f:
        for c in chunks:
            if written >= args.n:
                break
            raw = rag.llm.complete(
                [{"role": "user", "content": PROMPT.format(source=c.source, page=c.page, text=c.text)}],
                temperature=0.3,
            )
            try:
                item = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
            except ValueError:
                continue
            if not item.get("question"):
                continue
            item.update({"chunk_id": c.id, "doc_id": c.doc_id, "source": c.source, "page": c.page})
            f.write(json.dumps(item) + "\n")
            written += 1
            print(f"[{written}/{args.n}] {item['question']}")
    print(f"Wrote {written} questions to {args.out}")


if __name__ == "__main__":
    main()
