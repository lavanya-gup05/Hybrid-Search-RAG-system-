"""Load documents and split them into overlapping, sentence-aware chunks.

Supports .txt, .md and .pdf. Each chunk keeps its source file and a char span
so a citation can be traced back to an exact location in the original file.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, asdict
from typing import List

from .config import CFG

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    start: int
    end: int
    ordinal: int

    def to_dict(self):
        return asdict(self)


def read_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, source: str) -> List[Chunk]:
    """Greedy sentence packing up to chunk_size, with a character overlap so a
    claim that straddles a boundary still lands whole in at least one chunk."""
    text = clean(text)
    sentences = SENT_SPLIT.split(text)

    chunks: List[Chunk] = []
    buf, buf_start, cursor, ordinal = "", 0, 0, 0

    for sent in sentences:
        # locate the sentence in the original string to keep true offsets
        idx = text.find(sent, cursor)
        if idx == -1:
            idx = cursor
        cursor = idx + len(sent)

        if not buf:
            buf, buf_start = sent, idx
        elif len(buf) + len(sent) + 1 <= CFG.chunk_size:
            buf = f"{buf} {sent}"
        else:
            chunks.append(
                Chunk(str(uuid.uuid4())[:8], buf, source, buf_start,
                      buf_start + len(buf), ordinal)
            )
            ordinal += 1
            tail = buf[-CFG.chunk_overlap:] if CFG.chunk_overlap else ""
            buf = f"{tail} {sent}".strip()
            buf_start = max(0, idx - len(tail))

    if buf.strip():
        chunks.append(
            Chunk(str(uuid.uuid4())[:8], buf, source, buf_start,
                  buf_start + len(buf), ordinal)
        )
    return chunks


def load_corpus(paths: List[str]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith((".txt", ".md", ".pdf")):
                        fp = os.path.join(root, fn)
                        chunks += chunk_text(read_file(fp), os.path.basename(fp))
        else:
            chunks += chunk_text(read_file(p), os.path.basename(p))
    return chunks