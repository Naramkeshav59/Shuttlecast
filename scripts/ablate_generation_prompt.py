"""A/B the generation prompt on a saved result: keep every rally's agent
decisions (depth, suggestion_type) and re-run only generate_analysis with
the current prompt, so the prompt is the only variable. Writes the new
result over the old one, keeps the old one as *.before.json, and prints
tactical accuracy for both.

    python3 scripts/ablate_generation_prompt.py <match_id>
"""
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from groq import Groq  # noqa: E402

from shared.models import JobResult  # noqa: E402
from worker.agent.tools import GROQ_MODEL, _parse_pattern_suggestion, build_generation_prompt  # noqa: E402
from worker.eval.metrics import tactical_accuracy  # noqa: E402
from worker.pipeline.ingest import load_match, summarize_strokes  # noqa: E402


def main() -> None:
    match_id = sys.argv[1]
    path = REPO_ROOT / "data" / "results" / f"{match_id}.json"
    before = JobResult(**json.loads(path.read_text(encoding="utf-8")))
    rallies = {(r.set_num, r.rally_id): r for r in load_match(match_id)}
    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=6)

    updated, history = [], []
    prev_score = (0, 0)
    for report in before.rallies:
        a = report.analysis
        if a.depth != "skip":
            context = {
                "match_id": match_id, "score_a": prev_score[0], "score_b": prev_score[1],
                "rallies_analyzed": len(history), "recent_suggestions": history[-3:],
            }
            prompt = build_generation_prompt(
                summarize_strokes(rallies[(a.set_num, a.rally_id)].strokes), context, a.depth, a.suggestion_type,
            )
            text = client.chat.completions.create(
                model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
            ).choices[0].message.content
            pattern, suggestion = _parse_pattern_suggestion(text)
            a = a.model_copy(update={"tactical_pattern": pattern, "suggestion_text": suggestion})
            history.append(suggestion)
            print(f"  rally {a.rally_id}: {pattern[:80]}", file=sys.stderr)
        updated.append(report.model_copy(update={"analysis": a}))
        prev_score = (report.score_a, report.score_b)

    after = before.model_copy(update={"rallies": updated})
    shutil.copy(path, path.with_suffix(".before.json"))
    path.write_text(after.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({
        "before": tactical_accuracy(before.rallies),
        "after": tactical_accuracy(after.rallies),
    }, indent=2))


if __name__ == "__main__":
    main()
