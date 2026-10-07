"""Download BWF match clips referenced in ShuttleSet's match.csv via yt-dlp.

match_id is the `video` column from match.csv (the same folder name used
under data/shuttleset/set/) — that's the join key ingest.py will use to
find both the video file and the stroke CSVs for a match.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yt_dlp

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCH_CSV = REPO_ROOT / "data" / "shuttleset" / "set" / "match.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "videos"

# Non-commercial research use only
# fps<=31: a 60fps variant would break ShuttleSet's frame_num alignment
# (annotations are 25 or 30fps depending on the match).
YDL_FORMAT = (
    "bestvideo[height<=720][fps<=31][ext=mp4]+bestaudio[ext=m4a]"
    "/best[height<=720][fps<=31][ext=mp4]"
)


def load_matches() -> pd.DataFrame:
    return pd.read_csv(MATCH_CSV)


def select_matches(df: pd.DataFrame, match_ids: list[str] | None, count: int) -> pd.DataFrame:
    # a handful of ShuttleSet matches (e.g. id 12) have no linked BWF video
    df = df[df["url"].notna()]
    if match_ids:
        selected = df[df["video"].isin(match_ids)]
        missing = set(match_ids) - set(selected["video"])
        if missing:
            print(f"warning: match_id(s) not found (or missing url) in match.csv: {sorted(missing)}", file=sys.stderr)
        return selected
    return df.sort_values("duration").head(count)


def download_one(match_id: str, url: str, output_dir: Path, browser: str | None) -> None:
    dest = output_dir / f"{match_id}.mp4"
    if dest.exists():
        print(f"skip (already downloaded): {match_id}")
        return

    ydl_opts = {
        "format": YDL_FORMAT,
        "outtmpl": str(dest),
        "merge_output_format": "mp4",
        "quiet": False,
        "noprogress": False,
        # deno isn't installed on this machine; node is, and yt-dlp needs a
        # JS runtime to solve YouTube's signature cipher or downloads 403.
        "js_runtimes": {"node": {}},
    }
    if browser:
        # YouTube 403s anonymous format URLs after a few seconds of data;
        # authenticating as a logged-in browser session avoids that. The
        # android_vr client (yt-dlp's default fallback) ignores cookies, so
        # force the web client, which is the one that actually uses them.
        ydl_opts["cookiesfrombrowser"] = (browser,)
        ydl_opts["extractor_args"] = {"youtube": {"player_client": ["web"]}}
    print(f"downloading: {match_id} <- {url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        print(f"failed: {match_id}: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--match-ids", type=str, default=None,
        help="comma-separated video folder names (match_id) to download; overrides --count",
    )
    parser.add_argument(
        "--count", type=int, default=3,
        help="download the N shortest matches by duration (default: 3, ignored if --match-ids given)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--list", action="store_true",
        help="print selected matches (id, video, duration, url) without downloading",
    )
    parser.add_argument(
        "--browser", type=str, default=None,
        choices=["chrome", "edge", "firefox", "brave", "opera", "vivaldi", "safari", "chromium"],
        help="authenticate as this browser's logged-in YouTube session (works around anti-bot 403s)",
    )
    args = parser.parse_args()

    df = load_matches()
    match_ids = [m.strip() for m in args.match_ids.split(",")] if args.match_ids else None
    selected = select_matches(df, match_ids, args.count)

    if selected.empty:
        print("no matches selected", file=sys.stderr)
        sys.exit(1)

    if args.list:
        print(selected[["id", "video", "duration", "url"]].to_string(index=False))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in selected.iterrows():
        download_one(row["video"], row["url"], args.output_dir, args.browser)


if __name__ == "__main__":
    main()