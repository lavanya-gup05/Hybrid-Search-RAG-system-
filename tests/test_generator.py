"""If sentence splitting or citation parsing is wrong, the verifier checks the
wrong text against the wrong passage — it would silently mark good answers as
unsupported, or worse, let bad ones through. This is worth pinning down.
"""

from src.generator import split_sentences, cited_indices, format_context
from src.ingest import Chunk


def test_split_sentences_handles_multiple_citations():
    answer = ("The X400 carries a 24 month warranty [1]. The filter must be "
             "changed before 500 hours [2][3]. Warranty also covers paint.")
    sentences = split_sentences(answer)
    assert len(sentences) == 3
    assert sentences[0].startswith("The X400")
    assert sentences[1].startswith("The filter")


def test_cited_indices_extracts_all_numbers_in_order():
    assert cited_indices("Some claim [2][4].") == [2, 4]
    assert cited_indices("Some claim [4][2].") == [2, 4]  # sorted, not source order
    assert cited_indices("No citation here.") == []


def test_cited_indices_deduplicates():
    assert cited_indices("Claim cited twice [1][1].") == [1]


def test_format_context_numbers_passages_starting_at_one():
    chunks = [
        Chunk("a1", "First passage text.", "doc.md", 0, 20, 0),
        Chunk("a2", "Second passage text.", "doc.md", 20, 41, 1),
    ]
    ctx = format_context(chunks)
    assert "[1]" in ctx
    assert "[2]" in ctx
    assert "First passage text." in ctx
    assert "Second passage text." in ctx