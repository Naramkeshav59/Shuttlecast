"""The three independent evaluation layers.

1. Tactical accuracy -- does the analysis blame the player who actually
   lost the rally? Checkable against ShuttleSet's rally_winner with no
   human labels.
2. Suggestion quality -- cosine similarity of sentence embeddings between
   generated and reference suggestions (target: mean >= 0.75).
3. Visual-stroke consistency -- does a vision model's shot classification
   of a stroke frame agree with ShuttleSet's shot_type? Catches
   hallucination in the visual channel.

    python3 -m worker.eval.metrics data/results/<match_id>.json
"""
import json
import re
import sys
from collections.abc import Callable, Sequence

from shared.models import JobResult, RallyReport

# "Player A", "A's"/"A’s", "(A)". Deliberately not a bare "A": that would
# match the article in "A short clear by B...".
_BLAME = re.compile(r"\bPlayer\s+([AB])\b|\b([AB])['’]s\b|\(([AB])\)")


def blamed_player(tactical_pattern: str | None) -> str | None:
    """First player the pattern attributes the error to, if any."""
    if not tactical_pattern:
        return None
    m = _BLAME.search(tactical_pattern)
    return next(g for g in m.groups() if g) if m else None


def tactical_accuracy(reports: Sequence[RallyReport]) -> dict:
    """Share of analyzed (non-skip) rallies whose pattern blames the loser."""
    correct = scored = unattributed = 0
    for r in reports:
        if r.analysis.depth == "skip" or r.rally_winner is None:
            continue
        blamed = blamed_player(r.analysis.tactical_pattern)
        if blamed is None:
            unattributed += 1
            continue
        scored += 1
        correct += blamed != r.rally_winner
    return {
        "accuracy": correct / scored if scored else None,
        "scored": scored,
        "unattributed": unattributed,
    }


def _default_embedder() -> Callable[[list[str]], list[list[float]]]:
    from sentence_transformers import SentenceTransformer  # heavy; eval-only dependency

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return lambda texts: model.encode(texts, normalize_embeddings=True).tolist()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def suggestion_similarity(
    generated: Sequence[str],
    references: Sequence[str],
    embed: Callable[[list[str]], list[list[float]]] | None = None,
    threshold: float = 0.75,
) -> dict:
    if len(generated) != len(references):
        raise ValueError("generated and references must be aligned 1:1")
    if not generated:
        return {"mean": None, "pct_above_threshold": None, "n": 0}
    embed = embed or _default_embedder()
    gen_vecs = embed(list(generated))
    ref_vecs = embed(list(references))
    sims = [_cosine(g, r) for g, r in zip(gen_vecs, ref_vecs)]
    return {
        "mean": sum(sims) / len(sims),
        "pct_above_threshold": sum(s >= threshold for s in sims) / len(sims),
        "n": len(sims),
    }


# 18 ShuttleSet shot types -> 7 coarse families. Telling a "passive drop"
# from a "drop" in one still frame is unrealistic; the family is the fair test.
SHOT_FAMILY = {
    "short service": "serve", "long service": "serve",
    "net shot": "net", "return net": "net", "cross-court net shot": "net", "rush": "net",
    "lob": "lift", "defensive return lob": "lift",
    "clear": "clear",
    "smash": "smash", "wrist smash": "smash",
    "drop": "drop", "passive drop": "drop",
    "drive": "drive", "driven flight": "drive", "back-court drive": "drive",
    "push": "drive", "defensive return drive": "drive",
}


def _norm(shot: str) -> str:
    # models write "cross court", "cross-court", "cross‑court" (U+2011) interchangeably
    return re.sub(r"[\s\-‑]+", " ", shot.strip().lower())


_FAMILY = {_norm(k): v for k, v in SHOT_FAMILY.items()}


def visual_stroke_consistency(pairs: Sequence[tuple[str, str]]) -> dict:
    """pairs = (vision model's predicted shot, ShuttleSet shot_type).
    Strokes ShuttleSet itself marks 'unknown' are excluded."""
    pairs = [(_norm(p), _norm(t)) for p, t in pairs if _norm(t) != "unknown"]
    if not pairs:
        return {"exact": None, "family": None, "n": 0}
    exact = sum(p == t for p, t in pairs)
    family = sum(_FAMILY.get(p, p) == _FAMILY.get(t, t) for p, t in pairs)
    return {"exact": exact / len(pairs), "family": family / len(pairs), "n": len(pairs)}


def main() -> None:
    result = JobResult(**json.loads(open(sys.argv[1], encoding="utf-8").read()))
    print(json.dumps({"tactical_accuracy": tactical_accuracy(result.rallies)}, indent=2))


if __name__ == "__main__":
    main()
