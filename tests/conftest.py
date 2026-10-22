import io
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import RAGPipeline, Settings  # noqa: E402
from rag.embeddings import HashEmbedder  # noqa: E402


class FakeLLM:
    """Echoes the first source back with a citation; good enough to test plumbing offline."""

    def complete(self, messages, temperature=0.0):
        content = messages[-1]["content"]
        if content.startswith("Rewrite the user's latest message"):
            return re.search(r"Latest message: (.*)\n", content).group(1) + " (standalone)"
        return "According to the documents, see [1]. Also [9]."

    def stream(self, messages, temperature=0.0):
        yield from ["According ", "to ", "[1]."]


def make_pdf(pages: list[str]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def pipeline(tmp_path):
    s = Settings(data_dir=tmp_path, collection="test", chunk_size=300, chunk_overlap=50, top_k=3, use_reranker=False)
    return RAGPipeline(s, embedder=HashEmbedder(), llm=FakeLLM())


@pytest.fixture
def sample_pdf():
    return make_pdf(
        [
            "Photosynthesis converts sunlight, water and carbon dioxide into glucose.\n"
            "It happens in the chloroplasts of plant cells.",
            "The mitochondria is the powerhouse of the cell.\n"
            "It produces ATP through cellular respiration.",
            "Error code E-4012 means the payment gateway timed out.\n"
            "Retry the transaction after thirty seconds.",
        ]
    )
