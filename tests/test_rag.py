from rag.generator import cited_indices
from rag.ingest import Chunk, split_text
from rag.retriever import reciprocal_rank_fusion
from rag.store import Hit


def test_split_respects_size_and_overlaps():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = split_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share some words
    assert set(chunks[0].split()[-3:]) & set(chunks[1].split()[:10])


def test_rrf_rewards_agreement():
    def h(cid):
        return Hit(Chunk(cid, "d", "s", 1, 0, cid), 0.0)

    fused = reciprocal_rank_fusion([[h("a"), h("b"), h("c")], [h("b"), h("d"), h("a")]])
    assert [x.chunk.id for x in fused][:2] in (["a", "b"], ["b", "a"])
    assert fused[-1].chunk.id in {"c", "d"}


def test_cited_indices_filters_out_of_range():
    assert cited_indices("x [2] y [1][2] z [7]", 3) == [2, 1]


def test_ingest_is_idempotent(pipeline, sample_pdf):
    r1 = pipeline.ingest_pdf(sample_pdf, "bio.pdf")
    assert r1.pages == 3 and r1.chunks >= 3 and not r1.skipped
    r2 = pipeline.ingest_pdf(sample_pdf, "bio-copy.pdf")
    assert r2.skipped
    docs = pipeline.list_docs()
    assert len(docs) == 1 and docs[0]["source"] == "bio.pdf"


def test_bm25_finds_exact_codes(pipeline, sample_pdf):
    pipeline.ingest_pdf(sample_pdf, "bio.pdf")
    hits = pipeline.retriever.retrieve("what does E-4012 mean", top_k=1, mode="bm25")
    assert hits[0].chunk.page == 3


def test_hybrid_and_ask(pipeline, sample_pdf):
    pipeline.ingest_pdf(sample_pdf, "bio.pdf")
    hits = pipeline.retriever.retrieve("powerhouse of the cell mitochondria", top_k=2)
    assert hits[0].chunk.page == 2
    ans = pipeline.ask("What does the mitochondria do?")
    assert ans.sources and ans.cited == [1]  # [9] is out of range and dropped


def test_followup_is_condensed(pipeline, sample_pdf):
    pipeline.ingest_pdf(sample_pdf, "bio.pdf")
    hist = [{"role": "user", "content": "tell me about chloroplasts"}, {"role": "assistant", "content": "..."}]
    ans = pipeline.ask("where does it happen?", history=hist)
    assert ans.search_query.endswith("(standalone)")


def test_doc_filter_and_delete(pipeline, sample_pdf):
    from tests.conftest import make_pdf

    pipeline.ingest_pdf(sample_pdf, "bio.pdf")
    other = pipeline.ingest_pdf(make_pdf(["Mitochondria appear in a second document too."]), "other.pdf")
    hits = pipeline.retriever.retrieve("mitochondria", top_k=5, doc_ids=[other.doc_id])
    assert hits and all(h.chunk.doc_id == other.doc_id for h in hits)
    pipeline.delete_doc(other.doc_id)
    assert [d["source"] for d in pipeline.list_docs()] == ["bio.pdf"]
    assert all(h.chunk.source == "bio.pdf" for h in pipeline.retriever.retrieve("mitochondria", top_k=5))


def test_stream(pipeline, sample_pdf):
    pipeline.ingest_pdf(sample_pdf, "bio.pdf")
    q, hits, stream = pipeline.ask_stream("photosynthesis")
    assert hits and "".join(stream) == "According to [1]."


def test_rerank_reorders_candidates(pipeline, sample_pdf, monkeypatch):
    import rag.retriever as R

    class FakeCrossEncoder:
        """Scores a passage high only if it mentions 'payment' — the opposite of what hybrid ranks first."""

        def predict(self, pairs):
            return [10.0 if "payment" in passage.lower() else 0.0 for _, passage in pairs]

    monkeypatch.setattr(R, "_load_cross_encoder", lambda name: FakeCrossEncoder())
    pipeline.ingest_pdf(sample_pdf, "bio.pdf")
    q = "mitochondria powerhouse of the cell"
    assert pipeline.retriever.retrieve(q, top_k=1)[0].chunk.page == 2
    reranked = pipeline.retriever.retrieve(q, top_k=1, rerank=True)
    assert reranked[0].chunk.page == 3 and reranked[0].score == 10.0


def test_rerank_is_on_by_default(monkeypatch, tmp_path):
    from rag import Settings

    monkeypatch.delenv("USE_RERANKER", raising=False)
    assert Settings(data_dir=tmp_path).use_reranker is True
    monkeypatch.setenv("USE_RERANKER", "false")
    assert Settings(data_dir=tmp_path).use_reranker is False


def test_rerank_only_scores_the_top_candidates(pipeline, sample_pdf, monkeypatch):
    import rag.retriever as R

    seen = []

    class CountingCrossEncoder:
        def predict(self, pairs):
            seen.append(len(pairs))
            return [0.0] * len(pairs)

    monkeypatch.setattr(R, "_load_cross_encoder", lambda name: CountingCrossEncoder())
    pipeline.ingest_pdf(sample_pdf, "bio.pdf")
    total = len(pipeline.retriever.retrieve("the cell", top_k=50))
    assert total > 2
    pipeline.retriever.retrieve("the cell", top_k=5, rerank=True, rerank_candidates=2)
    pipeline.retriever.retrieve("the cell", top_k=5, rerank=True, rerank_candidates=0)  # 0 = all
    assert seen == [2, total]


def test_needs_rewrite_only_for_follow_ups():
    from rag.generator import needs_rewrite

    hist = [{"role": "user", "content": "How does SmartRecco use ChromaDB?"}, {"role": "assistant", "content": "..."}]
    assert not needs_rewrite("Why is that better?", None)  # first question: never
    assert needs_rewrite("Why is that better than traditional systems?", hist)
    assert needs_rewrite("and the backend?", hist)  # short
    assert needs_rewrite("Where does it run?", hist)
    assert not needs_rewrite("What are the hardware and software requirements to run the system?", hist)
    assert not needs_rewrite("What was the project's total budget in rupees?", hist)


def test_self_contained_follow_up_skips_the_rewrite_call(pipeline, sample_pdf):
    pipeline.ingest_pdf(sample_pdf, "bio.pdf")
    hist = [{"role": "user", "content": "tell me about chloroplasts"}, {"role": "assistant", "content": "..."}]
    ans = pipeline.ask("What does error code E-4012 mean for payments?", history=hist)
    assert ans.search_query == "What does error code E-4012 mean for payments?"


def test_prompt_repeats_citation_rule_at_the_end_and_trims_history():
    from rag.generator import ANSWER_REMINDER, SYSTEM_PROMPT, build_messages
    from rag.ingest import Chunk
    from rag.store import Hit

    hits = [Hit(Chunk("a-0", "a", "r.pdf", 3, 0, "Uses Django."), 1.0)]
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 2000} for i in range(10)]
    msgs = build_messages("What backend?", hits, history)
    assert msgs[0]["content"] == SYSTEM_PROMPT and "[2]" in SYSTEM_PROMPT  # has a worked example
    assert msgs[-1]["content"].endswith(ANSWER_REMINDER)
    assert "[1] (source: r.pdf, page 3)" in msgs[-1]["content"]
    assert len(msgs) == 1 + 4 + 1 and all(len(m["content"]) <= 600 for m in msgs[1:-1])


def test_llm_answers_the_rewritten_follow_up(pipeline, sample_pdf):
    pipeline.ingest_pdf(sample_pdf, "bio.pdf")
    seen = []
    real_complete = pipeline.llm.complete

    def spy(messages, temperature=0.0):
        seen.append(messages[-1]["content"])
        return real_complete(messages, temperature)

    pipeline.llm.complete = spy
    hist = [{"role": "user", "content": "tell me about chloroplasts"}, {"role": "assistant", "content": "..."}]
    pipeline.ask("where does it happen?", history=hist)
    answer_prompt = seen[-1]
    assert "Question: where does it happen? (standalone)" in answer_prompt
