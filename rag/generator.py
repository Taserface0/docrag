"""Answer generation with inline [n] citations, plus follow-up question condensing."""
from __future__ import annotations

import re
import json
import urllib.error
import urllib.request
from typing import Iterator, Protocol

from .store import Hit

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.

Rules:
- Put the source number in square brackets right after each fact it supports, e.g. "... uses Django [2]."
- If the sources do not contain the answer, say "I couldn't find that in the uploaded documents." Do not guess.
- Be concise: a few sentences, or short bullet points when listing several items.
- Do not use headings. Do not start with phrases like "[1] indicates" or "According to the sources".
- Never cite a source number that was not provided.

Example
Sources:
[1] (source: report.pdf, page 4) The backend is built with Django and exposes a REST API.
[2] (source: report.pdf, page 9) The application is deployed on AWS EC2.
Question: What is the backend and where is it hosted?
Answer: The backend is built with Django and exposes a REST API [1]. It is hosted on AWS EC2 [2]."""

# Repeated at the end of every request: small local models follow the most recent
# instruction far more reliably than one buried at the top of a long prompt.
ANSWER_REMINDER = (
    "Answer using only the sources above. Put the source number in brackets right after "
    "each fact, like [1]. No headings. If the answer isn't in the sources, say "
    "\"I couldn't find that in the uploaded documents.\""
)

CONDENSE_PROMPT = """Rewrite the user's latest message as a standalone search query, resolving pronouns \
and references using the conversation. Output only the rewritten question.

Conversation:
{history}

Latest message: {question}
Standalone question:"""


class LLM(Protocol):
    def complete(self, messages: list[dict], temperature: float = 0.0) -> str: ...
    def stream(self, messages: list[dict], temperature: float = 0.0) -> Iterator[str]: ...


class OllamaChat:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, messages: list[dict], temperature: float = 0.0) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama chat request failed ({error.code}): {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not connect to Ollama at {self.base_url}: {error.reason}") from error
        return result["message"]["content"]

    def stream(self, messages: list[dict], temperature: float = 0.0) -> Iterator[str]:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama chat request failed ({error.code}): {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not connect to Ollama at {self.base_url}: {error.reason}") from error

        with response:
            for line in response:
                if line.strip():
                    event = json.loads(line)
                    if event.get("error"):
                        raise RuntimeError(f"Ollama chat request failed: {event['error']}")
                    content = event.get("message", {}).get("content")
                    if content:
                        yield content


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        blocks.append(f"[{i}] (source: {c.source}, page {c.page})\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Only the last 2 turns, trimmed: enough for conversational tone, and every extra
    # word is text a CPU model has to read before it can start answering.
    for m in (history or [])[-4:]:
        msgs.append({"role": m["role"], "content": m["content"][:600]})
    msgs.append(
        {
            "role": "user",
            "content": f"Sources:\n\n{format_context(hits)}\n\nQuestion: {question}\n\n{ANSWER_REMINDER}",
        }
    )
    return msgs


# Words that usually point back to something earlier in the conversation.
_REFERRING = re.compile(
    r"\b(it|its|it's|this|that|these|those|they|them|their|theirs|he|she|him|her|his|"
    r"there|above|previous|earlier|former|latter|same|such|also|else|more)\b",
    re.IGNORECASE,
)


def needs_rewrite(question: str, history: list[dict] | None) -> bool:
    """Is this a follow-up that only makes sense with the chat history?

    Rewriting costs a whole extra LLM call, so skip it for self-contained questions.
    Short questions ("and the backend?") and ones with referring words ("why is that
    better?") are treated as follow-ups.
    """
    if not history:
        return False
    words = re.findall(r"[\w']+", question)
    return len(words) <= 4 or bool(_REFERRING.search(question))


def condense_question(llm: LLM, question: str, history: list[dict] | None) -> str:
    if not needs_rewrite(question, history):
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
    out = llm.complete([{"role": "user", "content": CONDENSE_PROMPT.format(history=convo, question=question)}])
    return out.strip() or question


def cited_indices(answer: str, n_sources: int) -> list[int]:
    """1-based source numbers the answer actually cites, in order of first appearance."""
    seen: list[int] = []
    for m in re.findall(r"\[(\d+)\]", answer):
        i = int(m)
        if 1 <= i <= n_sources and i not in seen:
            seen.append(i)
    return seen
