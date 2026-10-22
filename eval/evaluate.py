"""Evaluate retrieval (and optionally answers) against a test set.

Retrieval metrics per retriever configuration:
  hit@1, hit@k  — was the gold chunk retrieved in the top 1 / top k?
  MRR           — mean reciprocal rank of the gold chunk
  page@k        — was any chunk from the gold page retrieved? (lenient)

With --judge, also generates answers (hybrid) and asks the LLM to grade:
  faithfulness  — every claim supported by the retrieved sources?
  correctness   — does it agree with the reference answer?

    python -m eval.evaluate --testset eval/testset.jsonl --k 5 [--rerank [--rerank-candidates 0 20]] [--judge]
"""
from __future__ import annotations

import argparse
import json
import time

from rag import RAGPipeline
from rag.generator import format_context

JUDGE = """Grade a RAG system's answer.

Sources given to the system:
{context}

Question: {question}
Reference answer: {reference}
System answer: {answer}

Return JSON: {{"faithful": true/false, "correct": true/false, "reason": "<one sentence>"}}
- faithful: every factual claim in the system answer is supported by the sources.
- correct: the system answer agrees with the reference answer (wording may differ)."""


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieval_metrics(
    rag: RAGPipeline, items: list[dict], k: int, mode: str, rerank: bool, rerank_candidates: int | None = None
) -> dict:
    hit1 = hitk = mrr = pagek = 0.0
    t0 = time.perf_counter()
    for it in items:
        hits = rag.retriever.retrieve(
            it["question"], top_k=k, mode=mode, rerank=rerank, rerank_candidates=rerank_candidates
        )
        ids = [h.chunk.id for h in hits]
        if it["chunk_id"] in ids:
            r = ids.index(it["chunk_id"]) + 1
            hitk += 1
            hit1 += r == 1
            mrr += 1 / r
        pagek += any(h.chunk.doc_id == it["doc_id"] and h.chunk.page == it["page"] for h in hits)
    n = len(items)
    return {
        "retriever": mode + (_rerank_label(rerank_candidates) if rerank else ""),
        "hit@1": hit1 / n,
        f"hit@{k}": hitk / n,
        "MRR": mrr / n,
        f"page@{k}": pagek / n,
        "ms/query": 1000 * (time.perf_counter() - t0) / n,
    }


def _rerank_label(n: int | None) -> str:
    if n is None:
        return " + rerank"
    return " + rerank (all)" if n == 0 else f" + rerank (top {n})"


def judge_answers(rag: RAGPipeline, items: list[dict], k: int, rerank: bool) -> dict:
    faithful = correct = 0
    rows = []
    for it in items:
        a = rag.ask(it["question"], top_k=k, rerank=rerank)
        raw = rag.llm.complete(
            [{"role": "user", "content": JUDGE.format(
                context=format_context(a.sources), question=it["question"],
                reference=it["answer"], answer=a.text)}]
        )
        try:
            g = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        except ValueError:
            g = {"faithful": False, "correct": False, "reason": "unparseable judge output"}
        faithful += bool(g.get("faithful"))
        correct += bool(g.get("correct"))
        rows.append({**it, "system_answer": a.text, **g})
    n = len(items)
    return {"faithfulness": faithful / n, "correctness": correct / n, "rows": rows}


def print_table(rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    widths = [max(len(c), *(len(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(fmt(r[c]).ljust(w) for c, w in zip(cols, widths)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="eval/testset.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true", help="also evaluate hybrid + cross-encoder")
    ap.add_argument(
        "--rerank-candidates", type=int, nargs="+", default=[0, 20],
        help="how many fused candidates the reranker scores; one row per value (0 = all)",
    )
    ap.add_argument("--judge", action="store_true", help="also grade generated answers with the LLM")
    ap.add_argument("--out", default="eval/results.json")
    args = ap.parse_args()

    rag = RAGPipeline()
    items = load(args.testset)
    print(f"{len(items)} questions, k={args.k}\n")

    rows = [retrieval_metrics(rag, items, args.k, m, False) for m in ("dense", "bm25", "hybrid")]
    if args.rerank:
        for n in args.rerank_candidates:
            rows.append(retrieval_metrics(rag, items, args.k, "hybrid", True, rerank_candidates=n))
    print_table(rows)
    results = {"retrieval": rows}

    if args.judge:
        print("\nJudging answers…")
        j = judge_answers(rag, items, args.k, args.rerank)
        print(f"faithfulness: {j['faithfulness']:.3f}   correctness: {j['correctness']:.3f}")
        results["generation"] = j

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
