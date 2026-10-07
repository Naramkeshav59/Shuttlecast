"""Build the fine-tuning set: (decisive stroke frames + stroke data) -> analysis.

Labels come from a teacher model (Groq gpt-oss-120b, text-only) that sees
the full stroke data and the outcome; the student (Qwen2-VL-2B) learns to
produce the same analysis from frames + strokes -- a distillation setup.
Every label is marked needs_review: teacher output should be human-reviewed
before it's trusted as a reference.

Split by MATCH, not by rally: rallies of one match share players and
patterns, so a rally-level split would leak and inflate eval scores.

Resumable -- rallies already labeled are skipped, so hitting Groq's rate
limit just means re-running.

    python3 training/prepare_dataset.py [--max-per-match N]
Output: data/training/{train,eval}.jsonl and data/training/frames/
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from groq import Groq, RateLimitError  # noqa: E402

from worker.agent.tools import GROQ_MODEL  # noqa: E402
from worker.pipeline.frame_extractor import check_fps_alignment, extract_rally_frames  # noqa: E402
from worker.pipeline.inference import (  # noqa: E402
    format_target, parse_output, select_frames, user_text,
)
from worker.pipeline.ingest import expected_fps, load_match, summarize_strokes, verified_videos  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "training"
FRAMES_DIR = OUT_DIR / "frames"
EVAL_MATCHES = {"Evgeniya_Kosetskaya_Michelle_Li_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals"}

TEACHER_INSTRUCTIONS = """You are an elite badminton coach labeling training data. \
Below is one rally from a professional singles match. Decide:
- DEPTH: "skip" if the rally was won by a clean winner with no tactical error \
worth teaching; "surface" for a minor lesson; "deep" when a clear tactical \
weakness decided the rally.
- TYPE: positioning (court coverage / recovery), shot_selection (a better shot \
existed), or pattern_exploitation (a repeated pattern the opponent punished).
- PATTERN: one sentence naming the player who erred (as "A's ..." or "B's ...") \
and the specific stroke. The player who erred is almost always the rally's loser.
- SUGGESTION: a concrete correction for that player, specific to this rally. \
Avoid stock phrases like "tight net shot or crisp drive".

"""


def teacher_label(client: Groq, summary: dict, winner: str | None) -> str | None:
    prompt = TEACHER_INSTRUCTIONS + user_text(summary, winner)
    for _ in range(2):  # one retry if the teacher breaks format
        text = client.chat.completions.create(
            model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content
        parsed = parse_output(text)
        if parsed:
            return format_target(parsed["depth"], parsed["suggestion_type"],
                                 parsed["tactical_pattern"], parsed["suggestion_text"])
    return None


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-per-match", type=int, default=None)
    args = parser.parse_args()

    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "eval": 0, "skipped_bad_label": 0}

    for match_id, video in verified_videos().items():
        check_fps_alignment(video, expected_fps(match_id))
        split = "eval" if match_id in EVAL_MATCHES else "train"
        out_path = OUT_DIR / f"{split}.jsonl"
        done = load_done(out_path)
        rallies = load_match(match_id)[: args.max_per_match]
        print(f"{match_id}: {len(rallies)} rallies -> {split}", file=sys.stderr)

        with out_path.open("a", encoding="utf-8") as out:
            for rally in rallies:
                row_id = f"{match_id}/set{rally.set_num}/rally{rally.rally_id}"
                if row_id in done or not rally.strokes:
                    continue
                frames = extract_rally_frames(video, rally, frames_dir=FRAMES_DIR)
                summary = summarize_strokes(rally.strokes)
                try:
                    target = teacher_label(client, summary, rally.rally_winner)
                except RateLimitError as exc:
                    # Per-minute 429s are retried inside the client; a per-DAY
                    # quota can't be waited out in-process, so stop cleanly --
                    # everything written so far is kept and re-running resumes.
                    if "tokens per day" not in str(exc):
                        raise
                    print(json.dumps({**counts, "stopped": "Groq daily token quota reached; re-run later to resume"}))
                    return
                if target is None:
                    counts["skipped_bad_label"] += 1
                    continue
                picked = select_frames([frames[s.frame_num] for s in rally.strokes])
                out.write(json.dumps({
                    "id": row_id, "match_id": match_id,
                    "set_num": rally.set_num, "rally_id": rally.rally_id,
                    "rally_winner": rally.rally_winner,
                    # repo-relative so the dataset works after the trip to RunPod
                    "frames": [p.relative_to(REPO_ROOT).as_posix() for p in picked],
                    "stroke_summary": summary,
                    "target": target,
                    "teacher_model": GROQ_MODEL,
                    "needs_review": True,
                }) + "\n")
                out.flush()
                counts[split] += 1
                print(f"  {row_id}", file=sys.stderr)

    print(json.dumps(counts))


if __name__ == "__main__":
    main()
