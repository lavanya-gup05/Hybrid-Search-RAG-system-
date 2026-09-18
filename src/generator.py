"""Answer generation over the reranked context, with citations forced into the
output format so the verifier has something concrete to check."""

from __future__ import annotations

import re
from typing import List, Tuple

from groq import Groq, GroqError

from .config import CFG
from .ingest import Chunk

SYSTEM = """You answer strictly from the numbered CONTEXT passages provided.

Rules:
1. Every sentence that states a fact must end with one or more citations in the
   form [1] or [2][4], referring to the numbered passages.
2. Never state anything the passages do not contain. If the passages do not
   answer the question, reply exactly: INSUFFICIENT_CONTEXT
3. Do not cite a passage unless that specific passage supports that specific
   sentence. Do not pad sentences with extra citations.
4. Be concise. No preamble, no restating the question."""

CITE = re.compile(r"\[(\d+)\]")


def format_context(chunks: List[Chunk]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.source})\n{c.text}" for i, c in enumerate(chunks, 1)
    )


def split_sentences(answer: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", answer.strip())
    return [p.strip() for p in parts if p.strip()]


def cited_indices(sentence: str) -> List[int]:
    return sorted({int(m) for m in CITE.findall(sentence)})


class Generator:
    def __init__(self, api_key: str = CFG.groq_api_key, model: str = CFG.llm_model):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=api_key, timeout=CFG.llm_timeout_s)
        self.model = model

    def answer(self, question: str, chunks: List[Chunk]) -> str:
        ctx = format_context(chunks)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",
                     "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"},
                ],
            )
        except GroqError as e:
            raise RuntimeError(f"Generation failed (LLM provider error): {e}") from e
        return resp.choices[0].message.content.strip()