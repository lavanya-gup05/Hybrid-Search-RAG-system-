"""Citation verification — the layer most RAG demos skip.

An LLM that was told to cite will cite. That does not mean the cited passage
actually supports the sentence. This module re-checks each (claim, cited
passage) pair as an independent entailment judgement, with a fresh call that
never sees the question or the rest of the answer, so it cannot be dragged
along by the model's own narrative.

Verdicts per claim:
  SUPPORTED   - the passage states or directly entails the claim
  PARTIAL     - partly supported; some detail in the claim is not in the passage
  UNSUPPORTED - the passage does not support it
  UNCITED     - the sentence makes a factual claim but carries no citation
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from groq import Groq, GroqError

from .config import CFG
from .ingest import Chunk
from .generator import split_sentences, cited_indices, CITE

JUDGE_SYSTEM = """You are a strict citation checker. You are given a PASSAGE and
a CLAIM. Decide whether the passage, on its own, supports the claim.

Answer with exactly one word:
SUPPORTED   - the passage states the claim or directly entails it
PARTIAL     - the passage supports part of the claim but not all of it
UNSUPPORTED - the passage does not support the claim

Do not use outside knowledge. Plausible is not supported. Output one word only."""

RANK = {"UNSUPPORTED": 0, "UNCITED": 0, "PARTIAL": 1, "SUPPORTED": 2}


@dataclass
class ClaimCheck:
    sentence: str
    citations: List[int]
    verdict: str
    per_citation: dict


class Verifier:
    def __init__(self, api_key: str = CFG.groq_api_key, model: str = CFG.llm_model):
        self.client = Groq(api_key=api_key, timeout=CFG.llm_timeout_s)
        self.model = model

    def _judge(self, claim: str, passage: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=5,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",
                     "content": f"PASSAGE:\n{passage}\n\nCLAIM:\n{claim}"},
                ],
            )
        except GroqError:
            # Fail closed: if the judge call itself fails (timeout, rate
            # limit, outage) we must not let the claim through as SUPPORTED.
            return "UNSUPPORTED"
        out = resp.choices[0].message.content.strip().upper()
        for label in ("SUPPORTED", "PARTIAL", "UNSUPPORTED"):
            if label in out:
                return label
        return "UNSUPPORTED"

    def verify(self, answer: str, chunks: List[Chunk]) -> List[ClaimCheck]:
        checks: List[ClaimCheck] = []
        for sent in split_sentences(answer):
            idxs = cited_indices(sent)
            claim = re.sub(r"\s+([.,;:!?])", r"\1",
                           re.sub(r"\s{2,}", " ", CITE.sub("", sent))).strip()

            if not claim or len(claim) < 15:
                continue  # connective fragment, nothing to verify

            if not idxs:
                checks.append(ClaimCheck(sent, [], "UNCITED", {}))
                continue

            per = {}
            for i in idxs:
                if 1 <= i <= len(chunks):
                    per[i] = self._judge(claim, chunks[i - 1].text)
                else:
                    per[i] = "UNSUPPORTED"  # hallucinated citation number
            best = max(per.values(), key=lambda v: RANK[v])
            checks.append(ClaimCheck(sent, idxs, best, per))
        return checks


def apply_verdicts(checks: List[ClaimCheck], mode: str = "flag") -> str:
    """Rebuild the answer from verified claims.

    mode='strip'  -> drop anything not SUPPORTED (nothing unsourced survives)
    mode='flag'   -> keep it, marked, so the user sees what was questionable
    """
    out = []
    for c in checks:
        if c.verdict == "SUPPORTED":
            out.append(c.sentence)
        elif mode == "strip":
            continue
        elif c.verdict == "PARTIAL":
            out.append(f"{c.sentence} ⚠️*(partially supported)*")
        else:
            out.append(f"~~{c.sentence}~~ ❌*(unsupported — removed)*")
    return " ".join(out) if out else "INSUFFICIENT_CONTEXT"


def stats(checks: List[ClaimCheck]) -> dict:
    total = len(checks) or 1
    counts = {k: 0 for k in ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNCITED")}
    for c in checks:
        counts[c.verdict] += 1
    counts["unsourced_rate"] = round(
        (counts["UNSUPPORTED"] + counts["UNCITED"]) / total, 3)
    counts["claims"] = len(checks)
    return counts