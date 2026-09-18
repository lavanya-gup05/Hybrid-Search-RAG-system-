"""End-to-end pipeline: retrieve -> fuse -> rerank -> generate -> verify -> repair."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from .config import CFG
from .ingest import Chunk, load_corpus
from .retriever import HybridRetriever
from .reranker import Reranker
from .generator import Generator
from .verifier import Verifier, ClaimCheck, apply_verdicts, stats


@dataclass
class RAGResult:
    question: str
    raw_answer: str
    final_answer: str
    contexts: List[Chunk]
    rerank_scores: List[float]
    checks: List[ClaimCheck] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, retriever: HybridRetriever, reranker: Reranker | None = None,
                 generator: Generator | None = None, verifier: Verifier | None = None):
        self.retriever = retriever
        self.reranker = reranker or Reranker()
        self.generator = generator or Generator()
        self.verifier = verifier or Verifier()

    @classmethod
    def from_paths(cls, paths: List[str]) -> "Pipeline":
        chunks = load_corpus(paths)
        if not chunks:
            raise ValueError("No readable documents found.")
        return cls(HybridRetriever(chunks))

    def run(self, question: str, verify: bool = CFG.verify, mode: str = "flag",
            fused_top_k: int = CFG.fused_top_k,
            final_top_k: int = CFG.final_top_k) -> RAGResult:
        """fused_top_k/final_top_k are accepted as arguments (not read off the
        shared CFG singleton) so that per-user UI sliders in a multi-session
        deployment can't leak into other users' concurrent requests."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Question is empty.")
        if len(question) > CFG.max_question_chars:
            raise ValueError(
                f"Question is too long ({len(question)} chars, "
                f"max {CFG.max_question_chars}).")
        t = {}

        t0 = time.time()
        candidates = self.retriever.search(question, fused_top_k)
        t["retrieve"] = round(time.time() - t0, 2)

        t0 = time.time()
        ranked = self.reranker.rerank(question, candidates, final_top_k)
        t["rerank"] = round(time.time() - t0, 2)
        contexts = [c for c, _ in ranked]
        scores = [s for _, s in ranked]

        t0 = time.time()
        raw = self.generator.answer(question, contexts)
        t["generate"] = round(time.time() - t0, 2)

        if raw.strip() == "INSUFFICIENT_CONTEXT":
            return RAGResult(question, raw,
                             "I couldn't find support for this in the indexed "
                             "documents, so I'm not answering.",
                             contexts, scores, [], {}, t)

        if not verify:
            return RAGResult(question, raw, raw, contexts, scores, [], {}, t)

        t0 = time.time()
        checks = self.verifier.verify(raw, contexts)
        t["verify"] = round(time.time() - t0, 2)

        return RAGResult(question, raw, apply_verdicts(checks, mode),
                         contexts, scores, checks, stats(checks), t)
    