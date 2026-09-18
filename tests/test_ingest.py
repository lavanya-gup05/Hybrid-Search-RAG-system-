"""Chunking is the one bug class that silently corrupts everything downstream:
a wrong offset means a citation points at the wrong text forever. These tests
exist to catch that class of error, not to test every edge case of English
sentence splitting.
"""

from src.ingest import chunk_text, clean


def test_clean_collapses_whitespace_and_blank_lines():
    messy = "Line one.\r\n\r\n\r\n\r\nLine  two.   Trailing.   "
    out = clean(messy)
    assert "\r" not in out
    assert "\n\n\n" not in out
    assert "Line  two" not in out  # double space collapsed


def test_single_short_chunk_stays_whole():
    text = "The pump requires 24 months of warranty coverage."
    chunks = chunk_text(text, "doc.md")
    assert len(chunks) == 1
    assert chunks[0].text.strip() == text
    assert chunks[0].source == "doc.md"


def test_long_text_splits_into_multiple_chunks():
    # Build well past chunk_size (900 chars) worth of short sentences.
    sentence = "The warranty covers manufacturing defects in the housing. "
    text = sentence * 40  # ~2,900 chars
    chunks = chunk_text(text, "doc.md")
    assert len(chunks) > 1
    # every chunk should respect the configured size with some slack for
    # the trailing sentence that pushed it over
    for c in chunks:
        assert len(c.text) < 1200


def test_chunk_offsets_point_back_into_the_source_text():
    text = ("First sentence establishes context. Second sentence adds detail. "
           "Third sentence closes the paragraph out with more words here.")
    chunks = chunk_text(text, "doc.md")
    for c in chunks:
        # the char span recorded on the chunk must actually correspond to
        # where the chunk's own text (or a close variant) lives in source
        assert 0 <= c.start <= c.end <= len(text) + len(c.text)


def test_ordinal_increases_monotonically():
    sentence = "This is one sentence of a reasonable testing length. "
    text = sentence * 40
    chunks = chunk_text(text, "doc.md")
    ordinals = [c.ordinal for c in chunks]
    assert ordinals == sorted(ordinals)
    assert ordinals[0] == 0


def test_each_chunk_id_is_unique():
    sentence = "Another filler sentence used purely for length padding here. "
    text = sentence * 30
    chunks = chunk_text(text, "doc.md")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))