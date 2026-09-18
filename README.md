# Hybrid-Search RAG with Citation Verification

![tests](https://github.com/lavanya-gup05/Hybrid-Search-RAG-system-/actions/workflows/tests.yml/badge.svg)

A retrieval-augmented QA system that combines lexical and semantic retrieval,
reranks with a cross-encoder, and then **verifies every citation in the answer**
against the passage it points to before showing it to the user.

Most RAG demos stop at "retrieve top-k, stuff into prompt, print answer." This
one adds the two layers that decide whether the output is actually trustworthy:
better candidate ordering, and a check that the model's citations aren't
decorative.

## Pipeline

```
query
  ├── BM25 (lexical)      ──┐
  └── dense embeddings     ─┴── Reciprocal Rank Fusion → top 20
                                        │
                              cross-encoder rerank → top 5
                                        │
                              grounded generation (citations forced)
                                        │
                              per-claim citation verification
                                        │
                              flag or strip unsupported claims
```

### Why each piece exists

**BM25 + dense.** Embeddings blur rare literal tokens. Ask "what does error
E-217 mean" and a dense retriever happily returns the paragraph about E-218,
because the two are near-identical in embedding space. BM25 gets the exact token
right. Conversely, BM25 fails completely on paraphrase — "can freelancers
expense a client dinner" never matches the policy line that says "contractors
and agency staff are not eligible." Running both covers both failure modes.
Dense vectors live in ChromaDB rather than a plain in-memory array, so the
index survives a restart instead of being rebuilt from scratch every time.

**Reciprocal Rank Fusion.** BM25 scores are unbounded; cosine similarity sits in
[-1, 1]. Blending them with weights means tuning a constant that breaks on the
next corpus. RRF only uses *rank position*, so it needs no calibration:
`score(d) = Σ 1/(k + rank_i(d))`.

**Cross-encoder reranking.** A bi-encoder embeds query and document separately
and never compares their tokens. A cross-encoder reads the pair together and is
substantially more accurate — but too slow for the whole corpus. So: retrieve 20
cheaply, rerank those 20 precisely, keep 5.

**Citation verification.** An LLM told to cite will cite. That is not the same as
the cited passage supporting the sentence. Each claim is re-checked in an
isolated call that sees only the claim and the one passage — no question, no
rest of the answer — so the model can't be carried along by its own narrative.
Verdicts: `SUPPORTED` / `PARTIAL` / `UNSUPPORTED` / `UNCITED`. In *strip* mode
nothing unsourced reaches the user; in *flag* mode it's shown struck through so
you can see what the base model tried to slip past.

## Setup

### Option A — Docker (recommended, matches how you'd actually deploy this)

```bash
echo "GROQ_API_KEY=your_key_here" > .env
docker compose up --build
```

Open `http://localhost:8501`. First build takes a few minutes (downloading
torch + the embedding/reranking model weights); after that, both the Chroma
index and the cached model weights persist in named Docker volumes, so
rebuilds and restarts are fast.

### Option B — local Python

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...        # free key at console.groq.com
streamlit run app.py
```

First run downloads two small models (~120 MB total) from Hugging Face.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v
```

21 tests covering chunking (sentence packing, overlap, offset correctness),
RRF fusion ranking, citation parsing, and verdict/stats logic — the parts of
the pipeline that are wrong-and-silent if they break. Runs on every push via
GitHub Actions (badge above). The LLM-calling parts (`_judge`, `answer`) are
intentionally not unit-tested — that's what the eval harness and manual
spot-checks are for.

## Evaluation

```bash
python -m eval.run_eval --docs data --gold eval/gold.json          # full
python -m eval.run_eval --docs data --gold eval/gold.json --skip-llm  # retrieval only, no API
```

Prints recall@5 for BM25 / dense / hybrid / hybrid+rerank, and the rate of
unsourced claims the verifier catches. **Run this on your own corpus and put
the real numbers in your resume bullet** — measured numbers survive interview
questions, invented ones don't.

The sample corpus in `data/` is deliberately built with near-miss traps (E-217
vs E-218, X400 vs X300 warranty terms, contractors vs employees) so the ablation
actually separates the modes instead of everything scoring 1.0.

### Results

Measured on the sample corpus (3 docs, 8 chunks, 10 gold questions, deliberate
near-miss traps like E-217 vs E-218 and Business-tier vs Team-tier pricing):

| Retrieval mode | Recall@1 |
|---|---|
| BM25 only | 1.000 |
| Dense only | 0.800 |
| Hybrid (RRF) | 1.000 |
| Hybrid + rerank | 0.900 |

Recall@5 saturates at 1.000 across every mode on this corpus size — with only
8 total chunks, retrieving the top 5 returns most of the corpus regardless of
method, so recall@1 (does the *first* result match) is the metric that
actually separates them here. Dense-only retrieval missed the target passage
outright on 2 of 10 questions, mostly the near-miss traps written to look
similar in embedding space. Reranking underperformed plain hybrid fusion by
one question — worth investigating (the trap passages contain a lot of
qualifying detail, which cross-encoders trained on cleaner query-passage
pairs may weight oddly) rather than a result to hide.

Citation verification unsourced-claim rate: _run the full eval (below) with a
Groq key and paste the number here._

## Files

| File | What it does |
|---|---|
| `src/ingest.py` | Loads .pdf/.txt/.md, sentence-aware chunking with overlap and true char offsets |
| `src/retriever.py` | BM25 + ChromaDB dense search, RRF fusion, persistent index |
| `src/reranker.py` | Cross-encoder reranking of fused candidates |
| `src/generator.py` | Grounded generation with enforced `[n]` citations |
| `src/verifier.py` | Per-claim entailment checking, verdict application |
| `src/pipeline.py` | Orchestration + timings + metrics |
| `app.py` | Streamlit UI: answer, citation audit table, retrieved passages, trace |
| `eval/run_eval.py` | Ablation harness |
| `tests/` | Unit tests — chunking, fusion, citation parsing, verdict logic |
| `.github/workflows/tests.yml` | CI: runs the test suite on every push |
| `Dockerfile` / `docker-compose.yml` | Containerized deploy with persistent volumes for the index and model cache |

## Extensions worth doing next

- Swap the LLM judge for a local NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`)
  — cheaper, and lets you compare judge agreement.
- Repair loop: on `UNSUPPORTED`, re-query with the failed claim as the query and
  regenerate that sentence only.
- Query decomposition for multi-hop questions.
- Persist the index (`retriever.save()`) so restarts don't re-embed.

## Resume bullet

> Built a hybrid-search RAG system (BM25 + dense retrieval, RRF fusion,
> cross-encoder reranking) with a citation-verification layer that
> independently checks each generated claim against its source. On a gold
> eval set with deliberate near-miss distractors, hybrid fusion matched or
> exceeded single-method retrieval (recall@1: BM25 1.00, dense 0.80, hybrid
> 1.00), while reranking revealed a tradeoff worth further study (0.90).
> Containerized with Docker, tested with a 21-case pytest suite running in
> CI on every push.

Once you've run the full eval (with your Groq key) and have a real
unsourced-claim-rate number, add a second clause: *"...and the citation
verifier flagged **[Z]%** of generated claims as unsupported before they
reached the user."*
