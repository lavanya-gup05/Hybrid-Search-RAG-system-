"""Hybrid retrieval: BM25 (lexical) + dense embeddings (semantic), fused with
Reciprocal Rank Fusion.

Why both: BM25 nails rare literal tokens (error codes, product names, section
numbers) that embeddings smear away; dense retrieval catches paraphrase that
BM25 misses entirely. RRF merges the two ranked lists without needing the two
score scales to be comparable, which is the usual failure of naive weighted
score blending.

Dense storage uses ChromaDB (persisted to disk) rather than an in-memory
NumPy array. A plain array is fine for a demo but doesn't survive a restart
and doesn't scale — Chroma gives real persistence and indexed similarity
search with the same amount of code.
"""

from __future__ import annotations

import pickle
import re
from typing import Dict, List, Tuple

import chromadb
from rank_bm25 import BM25Okapi

from .config import CFG
from .ingest import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
COLLECTION_NAME = "chunks"


def tokenize(text: str) -> List[str]:
    return TOKEN.findall(text.lower())


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], embedder=None,
                 persist_dir: str = CFG.index_dir, rebuild: bool = True):
        self.chunks = chunks
        self.by_id: Dict[str, Chunk] = {c.id: c for c in chunks}

        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(CFG.embed_model)
        self.embedder = embedder

        self.client = chromadb.PersistentClient(path=persist_dir)

        if rebuild:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
            self._index(chunks)
        else:
            self.collection = self.client.get_collection(COLLECTION_NAME)

        with open(f"{persist_dir}/chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

    def _index(self, chunks: List[Chunk], batch_size: int = 64):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = self.embedder.encode(
                [c.text for c in batch],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()
            self.collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[{"source": c.source, "ordinal": c.ordinal} for c in batch],
            )

    # ---------- individual retrievers (kept separate so eval can ablate) -----

    def bm25_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1][:k]
        return [(self.chunks[i].id, float(scores[i])) for i in order]

    def dense_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True,
                                     normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=k)
        ids = res["ids"][0]
        # Chroma returns cosine *distance*; convert to a similarity score
        # (1 - distance) purely so higher-is-better matches bm25_search's
        # convention. Only relative order is used downstream (RRF), so the
        # exact scale doesn't matter.
        dists = res["distances"][0]
        return [(i, 1.0 - d) for i, d in zip(ids, dists)]

    # ---------- fusion ------------------------------------------------------

    @staticmethod
    def rrf(rankings: List[List[str]], k: int = CFG.rrf_k) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def search(self, query: str, top_k: int = CFG.fused_top_k) -> List[Chunk]:
        sparse = [d for d, _ in self.bm25_search(query, CFG.bm25_top_k)]
        dense = [d for d, _ in self.dense_search(query, CFG.dense_top_k)]
        fused = self.rrf([sparse, dense])[:top_k]
        return [self.by_id[doc_id] for doc_id, _ in fused]

    # ---------- persistence -------------------------------------------------
    # Chroma already persists embeddings to `persist_dir` as data is added, so
    # there's nothing extra to save there. `load` rebuilds BM25 (cheap, and
    # must stay in sync with the same chunk set) and reattaches to the
    # existing Chroma collection instead of re-embedding everything.

    @classmethod
    def load(cls, path: str = CFG.index_dir, embedder=None) -> "HybridRetriever":
        with open(f"{path}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        return cls(chunks, embedder=embedder, persist_dir=path, rebuild=False)
        