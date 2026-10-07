"""Load ShuttleSet stroke CSVs for a match into Rally/Stroke models.

Assumes the match's video has already been fetched (scripts/download_videos.py)
and its stroke data lives under data/shuttleset/set/{match_id}/set{N}.csv,
where match_id is the `video` column from match.csv.
"""
import re
from pathlib import Path

import pandas as pd

from shared.models import Rally, Stroke

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SHUTTLESET_DIR = REPO_ROOT / "data" / "shuttleset" / "set"

# ShuttleSet records shot type in Chinese; translation from the dataset's own
# README (18 shot types, BLSR representation).
SHOT_TYPE_TRANSLATION = {
    "放小球": "net shot",
    "擋小球": "return net",
    "殺球": "smash",
    "點扣": "wrist smash",
    "挑球": "lob",
    "防守回挑": "defensive return lob",
    "長球": "clear",
    "平球": "drive",
    "小平球": "driven flight",
    "後場抽平球": "back-court drive",
    "切球": "drop",
    "過渡切球": "passive drop",
    "推球": "push",
    "撲球": "rush",
    "防守回抽": "defensive return drive",
    "勾球": "cross-court net shot",
    "發短球": "short service",
    "發長球": "long service",
    "未知球種": "unknown",
    # ShuttleSet's own CSVs use this as an inconsistent typo of 過渡切球
    # (passive drop) in some matches -- not a distinct 19th shot type.
    "過度切球": "passive drop",
}


def _str_or_none(value) -> str | None:
    return None if pd.isna(value) else str(value)


def _float_or_none(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _row_to_stroke(rally_id: int, row: pd.Series) -> Stroke:
    raw_type = row["type"]
    shot_type = SHOT_TYPE_TRANSLATION.get(raw_type, raw_type)
    return Stroke(
        rally_id=rally_id,
        player=_str_or_none(row["player"]),
        ball_round=int(row["ball_round"]),
        frame_num=int(row["frame_num"]),
        shot_type=shot_type,
        hit_x=_float_or_none(row["hit_x"]),
        hit_y=_float_or_none(row["hit_y"]),
        landing_x=_float_or_none(row["landing_x"]),
        landing_y=_float_or_none(row["landing_y"]),
        player_location_x=_float_or_none(row["player_location_x"]),
        player_location_y=_float_or_none(row["player_location_y"]),
        opponent_location_x=_float_or_none(row["opponent_location_x"]),
        opponent_location_y=_float_or_none(row["opponent_location_y"]),
        getpoint_player=_str_or_none(row["getpoint_player"]),
    )


def load_set(match_id: str, set_num: int) -> list[Rally]:
    """Parse one set{N}.csv into Rally objects, grouped by rally number."""
    csv_path = SHUTTLESET_DIR / match_id / f"set{set_num}.csv"
    df = pd.read_csv(csv_path)

    rallies = []
    for rally_id, group in df.groupby("rally"):
        ordered = group.sort_values("ball_round")
        strokes = [_row_to_stroke(int(rally_id), row) for _, row in ordered.iterrows()]
        # getpoint_player is only populated on a rally's final stroke.
        rally_winner = strokes[-1].getpoint_player if strokes else None
        rallies.append(Rally(
            match_id=match_id,
            set_num=set_num,
            rally_id=int(rally_id),
            strokes=strokes,
            rally_winner=rally_winner,
        ))
    return rallies


def summarize_strokes(strokes: list[Stroke]) -> dict:
    """The stroke-level view of a rally. Shared by the agent's
    analyze_stroke_sequence tool and the fine-tuned model's prompt so both
    see identical data."""
    return {
        "rally_length": len(strokes),
        "shot_sequence": [s.shot_type for s in strokes],
        "players": [s.player for s in strokes],
        "landing_zones": [(s.landing_x, s.landing_y) for s in strokes],
        "player_positions": [(s.player_location_x, s.player_location_y) for s in strokes],
        "final_shot_type": strokes[-1].shot_type if strokes else None,
        "rally_winner": strokes[-1].getpoint_player if strokes else None,
    }


VIDEOS_DIR = REPO_ROOT / "data" / "videos"
VIDEO_REGISTRY = REPO_ROOT / "data" / "verified_videos.json"


def verified_videos() -> dict[str, Path]:
    """match_id -> local clip, only for clips confirmed to line up with
    ShuttleSet's frame_num (data/verified_videos.json) and present on disk."""
    import json

    registry = json.loads(VIDEO_REGISTRY.read_text(encoding="utf-8"))
    out = {}
    for match_id, entry in registry.items():
        if match_id.startswith("_") or not entry.get("aligned"):
            continue
        path = VIDEOS_DIR / entry["file"]
        if path.exists():
            out[match_id] = path
    return out


def expected_fps(match_id: str) -> float:
    """The fps ShuttleSet's frame_num was annotated at, recovered from
    frame_num / `time`. Not constant: 19 of the 44 matches are 25fps, the
    rest 30fps, so this can't be hardcoded."""
    df = pd.read_csv(SHUTTLESET_DIR / match_id / "set1.csv")
    seconds = pd.to_timedelta(df["time"]).dt.total_seconds()
    valid = seconds > 0
    return float((df.loc[valid, "frame_num"] / seconds[valid]).median())


def download_video(url: str, dest: Path) -> Path:
    """yt-dlp fetch of a match clip, video stream only (we only ever read
    frames, so no audio merge and no ffmpeg dependency). fps<=31 keeps
    YouTube from handing us a 60fps variant whose frame numbering would no
    longer line up with ShuttleSet's annotations."""
    import shutil

    import yt_dlp

    dest.parent.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestvideo[height<=720][fps<=31][ext=mp4]/best[height<=720][fps<=31][ext=mp4]",
        "outtmpl": str(dest),
        "quiet": True,
        "noprogress": True,
    }
    if shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return dest


def load_match(match_id: str) -> list[Rally]:
    """Load every set for a match, in set order."""
    match_dir = SHUTTLESET_DIR / match_id
    set_files = sorted(
        match_dir.glob("set*.csv"),
        key=lambda p: int(re.search(r"set(\d+)\.csv", p.name).group(1)),
    )
    rallies = []
    for f in set_files:
        set_num = int(re.search(r"set(\d+)\.csv", f.name).group(1))
        rallies.extend(load_set(match_id, set_num))
    return rallies
