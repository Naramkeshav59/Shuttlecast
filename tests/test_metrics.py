import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.models import AnalysisResult, RallyReport  # noqa: E402
from worker.eval.metrics import (  # noqa: E402
    blamed_player, suggestion_similarity, tactical_accuracy, visual_stroke_consistency,
)


@pytest.mark.parametrize("text,expected", [
    ("A’s third-shot clear was too short.", "A"),          # curly apostrophe, as gpt-oss writes it
    ("B's cross-court net shot was too high.", "B"),
    ("Player A failed to recover to base.", "A"),
    ("A short clear by B's opponent", "B"),                 # leading article "A" is not a player
    ("The rally ended on a smash.", None),
    (None, None),
])
def test_blamed_player(text, expected):
    assert blamed_player(text) == expected


def _report(depth, pattern, winner):
    analysis = AnalysisResult(match_id="m", set_num=1, rally_id=1, depth=depth, tactical_pattern=pattern)
    return RallyReport(analysis=analysis, start_time_sec=0, rally_winner=winner, n_strokes=4, score_a=0, score_b=0)


def test_tactical_accuracy_blames_the_loser():
    reports = [
        _report("deep", "B’s net shot was loose.", "A"),     # correct: B lost
        _report("surface", "A's clear was short.", "A"),     # wrong: blames the winner
        _report("deep", "The rally was long.", "B"),         # unattributed
        _report("skip", None, "A"),                          # skipped, not scored
    ]
    assert tactical_accuracy(reports) == {"accuracy": 0.5, "scored": 2, "unattributed": 1}


def test_suggestion_similarity_with_stub_embedder():
    vecs = {"x": [1.0, 0.0], "y": [1.0, 0.0], "z": [0.0, 1.0]}
    out = suggestion_similarity(["x", "x"], ["y", "z"], embed=lambda ts: [vecs[t] for t in ts])
    assert out == {"mean": 0.5, "pct_above_threshold": 0.5, "n": 2}


def test_visual_consistency_exact_vs_family_and_spelling():
    out = visual_stroke_consistency([
        ("Smash", "wrist smash"),            # family match only
        ("cross court net shot", "cross-court net shot"),
        ("back court drive", "back-court drive"),
        ("lob", "clear"),                    # different families
        ("smash", "unknown"),                # excluded
    ])
    assert out == {"exact": 0.5, "family": 0.75, "n": 4}
