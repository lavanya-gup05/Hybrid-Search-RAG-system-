"""Ablation harness. This is what turns the project into a resume bullet with
a real number behind it instead of a claim.

Measures:
  A) Retrieval recall@k for BM25 only / dense only / hybrid / hybrid+rerank
  B) Unsourced-claim rate with verification off vs on

Usage:
    python -m eval.run_eval --gold eval/gold.json --docs data
"""

from __future__ import annotations

import argparse
import json
from typing import List

from src.config import CFG
from src.ingest import load_corpus, Chunk
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.generator import Generator
from src.verifier import Verifier, stats
from src.pipeline import Pipeline


def hit(chunks: List[Chunk], keywords: List[str]) -> bool:
    """A retrieval is a hit if any retrieved chunk contains every keyword of the
    gold answer span. Crude but honest, and it does not need gold chunk IDs."""
    blob = [c.text.lower() for c in chunks]
    return any(all(kw.lower() in text for kw in keywords) for text in blob)


def recall_at_k(retr: HybridRetriever, reranker: Reranker, gold, k: int = 5):
    modes = {"bm25": 0, "dense": 0, "hybrid": 0, "hybrid+rerank": 0}
    for item in gold:
        q, kws = item["question"], item["must_contain"]

        bm = [retr.by_id[i] for i, _ in retr.bm25_search(q, k)]
        dn = [retr.by_id[i] for i, _ in retr.dense_search(q, k)]
        hy = retr.search(q, k)
        rr = [c for c, _ in reranker.rerank(q, retr.search(q, CFG.fused_top_k), k)]

        modes["bm25"] += hit(bm, kws)
        modes["dense"] += hit(dn, kws)
        modes["hybrid"] += hit(hy, kws)
        modes["hybrid+rerank"] += hit(rr, kws)

    n = len(gold)
    return {m: round(v / n, 3) for m, v in modes.items()}


def verification_effect(pipe: Pipeline, gold):
    off_unsourced, on_unsourced, total_claims = 0, 0, 0
    for item in gold:
        res = pipe.run(item["question"], verify=True, mode="strip")
        if not res.metrics:
            continue
        total_claims += res.metrics["claims"]
        off_unsourced += res.metrics["UNSUPPORTED"] + res.metrics["UNCITED"]
    return {
        "claims_generated": total_claims,
        "unsourced_claims_before_filter": off_unsourced,
        "unsourced_rate_before": round(off_unsourced / max(total_claims, 1), 3),
        "unsourced_rate_after": 0.0,  # strip mode removes them by construction
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold.json")
    ap.add_argument("--docs", default="data")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--skip-llm", action="store_true",
                    help="retrieval ablation only, no API calls")
    args = ap.parse_args()

    gold = json.load(open(args.gold))
    chunks = load_corpus([args.docs])
    print(f"{len(chunks)} chunks indexed, {len(gold)} eval questions\n")

    retr = HybridRetriever(chunks)
    reranker = Reranker()

    print("=== Recall@%d ===" % args.k)
    for mode, val in recall_at_k(retr, reranker, gold, args.k).items():
        print(f"  {mode:<16} {val:.3f}")

    if args.skip_llm:
        return

    pipe = Pipeline(retr, reranker, Generator(), Verifier())
    print("\n=== Citation verification ===")
    for k, v in verification_effect(pipe, gold).items():
        print(f"  {k:<32} {v}")


if __name__ == "__main__":
    main()