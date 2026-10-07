"""Run the full pipeline locally -- no AWS. Same analyze_match() the Fargate
worker runs; the result is written to data/results/{match_id}.json, which
the Streamlit UI can load in local mode.

    python3 scripts/local_run.py --match-id <id> [--max-rallies 10]
    python3 scripts/local_run.py --list
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from worker.agent.react_loop import analyze_match  # noqa: E402
from worker.pipeline.ingest import verified_videos  # noqa: E402

RESULTS_DIR = REPO_ROOT / "data" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id")
    parser.add_argument("--max-rallies", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="list matches with a verified local clip")
    args = parser.parse_args()

    videos = verified_videos()
    if args.list or not args.match_id:
        for match_id, path in videos.items():
            print(f"{match_id}  ({path.name})")
        return
    if args.match_id not in videos:
        sys.exit(f"no verified clip for {args.match_id}; see data/verified_videos.json")

    started = time.time()
    done = 0

    def progress(rally) -> None:
        nonlocal done
        done += 1
        print(f"  [{done}] set {rally.set_num} rally {rally.rally_id} ({time.time() - started:.0f}s)",
              file=sys.stderr)

    result = analyze_match(
        args.match_id, videos[args.match_id], max_rallies=args.max_rallies, on_rally_done=progress,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.match_id}.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"{len(result.rallies)} rallies analyzed, {len(result.errors)} failed -> {out}")


if __name__ == "__main__":
    main()
