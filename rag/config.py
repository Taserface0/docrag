"""Central configuration, read from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "qwen2.5:3b"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "nomic-embed-text"))
    # Task prefixes for the embedding model. Unset = use the model's recommended
    # prefixes (e.g. "search_document: " / "search_query: " for nomic-embed-text).
    embed_doc_prefix: str | None = field(default_factory=lambda: os.getenv("EMBED_DOC_PREFIX"))
    embed_query_prefix: str | None = field(default_factory=lambda: os.getenv("EMBED_QUERY_PREFIX"))
    embed_batch_size: int = field(default_factory=lambda: int(_env("EMBED_BATCH_SIZE", "32")))

    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")))
    collection: str = field(default_factory=lambda: _env("COLLECTION", "docrag"))

    # Chunking (characters). ~1000 chars ≈ 200–250 tokens.
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))

    # Retrieval
    candidates: int = field(default_factory=lambda: int(_env("CANDIDATES", "20")))  # per retriever
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))  # sent to the LLM
    use_reranker: bool = field(default_factory=lambda: _env("USE_RERANKER", "true").lower() == "true")
    rerank_candidates: int = field(default_factory=lambda: int(_env("RERANK_CANDIDATES", "20")))  # 0 = all
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
