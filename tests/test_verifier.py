"""These test the parts of verification that don't require a live LLM call:
turning ClaimChecks into a final answer string, and summarizing verdicts into
the metrics shown in the UI. The actual entailment judgement (_judge) calls
Groq and is intentionally not unit-tested here — that's what eval/run_eval.py
and manual spot-checking are for.
"""

from src.verifier import ClaimCheck, apply_verdicts, stats


def _check(sentence, verdict, citations=None):
    return ClaimCheck(sentence, citations or [1], verdict, {})


def test_apply_verdicts_flag_mode_keeps_everything_but_marks_bad_claims():
    checks = [
        _check("This is supported.", "SUPPORTED"),
        _check("This is not.", "UNSUPPORTED"),
    ]
    out = apply_verdicts(checks, mode="flag")
    assert "This is supported." in out
    assert "~~This is not.~~" in out


def test_apply_verdicts_strip_mode_removes_unsupported_entirely():
    checks = [
        _check("Keep me.", "SUPPORTED"),
        _check("Drop me.", "UNSUPPORTED"),
    ]
    out = apply_verdicts(checks, mode="strip")
    assert "Keep me." in out
    assert "Drop me." not in out


def test_apply_verdicts_all_unsupported_falls_back_to_insufficient_context():
    checks = [_check("Nothing backs this.", "UNCITED", citations=[])]
    out = apply_verdicts(checks, mode="strip")
    assert out == "INSUFFICIENT_CONTEXT"


def test_apply_verdicts_partial_gets_a_warning_marker_not_struck_through():
    checks = [_check("Half true.", "PARTIAL")]
    out = apply_verdicts(checks, mode="flag")
    assert "Half true." in out
    assert "~~" not in out
    assert "partially supported" in out


def test_stats_counts_each_verdict_bucket():
    checks = [
        _check("a", "SUPPORTED"),
        _check("b", "SUPPORTED"),
        _check("c", "PARTIAL"),
        _check("d", "UNSUPPORTED"),
        _check("e", "UNCITED", citations=[]),
    ]
    s = stats(checks)
    assert s["SUPPORTED"] == 2
    assert s["PARTIAL"] == 1
    assert s["UNSUPPORTED"] == 1
    assert s["UNCITED"] == 1
    assert s["claims"] == 5
    assert s["unsourced_rate"] == 0.4  # (1 unsupported + 1 uncited) / 5


def test_stats_on_empty_checks_does_not_divide_by_zero():
    s = stats([])
    assert s["claims"] == 0
    assert s["unsourced_rate"] == 0.0