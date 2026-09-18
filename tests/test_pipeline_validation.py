"""Pipeline.run should reject bad input before it burns a retrieval+LLM cycle.
No live retriever/LLM needed here — validation happens before either is touched.
"""

import pytest

from src.pipeline import Pipeline
from src.config import CFG


def _bare_pipeline():
    # Skip __init__ (which builds a real retriever/reranker/generator/verifier)
    # since we're only exercising the validation guard at the top of run().
    return Pipeline.__new__(Pipeline)


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        _bare_pipeline().run("   ")


def test_oversized_question_is_rejected():
    with pytest.raises(ValueError):
        _bare_pipeline().run("x" * (CFG.max_question_chars + 1))
