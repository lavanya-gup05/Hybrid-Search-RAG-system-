"""RRF is the piece that makes 'hybrid' retrieval actually hybrid rather than
'BM25 with a dense reranker bolted on.' If this ranks wrong, the whole
justification for running two retrievers collapses.
"""

from src.retriever import HybridRetriever


def test_document_ranked_first_by_both_retrievers_wins_fusion():
    sparse = ["a", "b", "c"]
    dense = ["a", "c", "b"]
    fused = HybridRetriever.rrf([sparse, dense])
    assert fused[0][0] == "a"


def test_document_missing_from_one_list_still_ranks():
    # 'x' appears only in dense results — hybrid fusion should still surface
    # it, just lower than documents both retrievers agree on.
    sparse = ["a", "b"]
    dense = ["x", "a", "b"]
    fused = HybridRetriever.rrf([sparse, dense])
    ids = [doc_id for doc_id, _ in fused]
    assert "x" in ids
    assert ids.index("a") < ids.index("x")  # agreement beats single-list rank 2


def test_fusion_score_is_higher_for_top_ranked_docs():
    sparse = ["a", "b", "c"]
    dense = ["a", "b", "c"]
    fused = dict(HybridRetriever.rrf([sparse, dense]))
    assert fused["a"] > fused["b"] > fused["c"]


def test_empty_rankings_returns_empty_list():
    assert HybridRetriever.rrf([[], []]) == []


def test_single_ranking_list_still_fuses_correctly():
    fused = HybridRetriever.rrf([["a", "b", "c"]])
    assert [doc_id for doc_id, _ in fused] == ["a", "b", "c"]