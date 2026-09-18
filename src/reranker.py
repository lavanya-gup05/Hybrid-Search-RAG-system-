"""Cross-encoder reranking.

Bi-encoders embed the query and the chunk separately, so they never actually
compare the two texts token-by-token. A cross-encoder scores the (query, chunk)
pair jointly, which is far more accurate but too slow to run over a whole
corpus. So: retrieve ~20 cheaply, rerank those 20 precisely, keep the top 5.
"""

from __future__ import annotations

from typing import List, Tuple

from .config import CFG
from .ingest import Chunk


class Reranker:
    def __init__(self, model_name: str = CFG.rerank_model):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Chunk],
               top_k: int = CFG.final_top_k) -> List[Tuple[Chunk, float]]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(chunks, map(float, scores)),
                        key=lambda x: x[1], reverse=True)
        return ranked[:top_k]