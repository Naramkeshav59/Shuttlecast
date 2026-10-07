"""Extract video frames at ShuttleSet stroke timestamps (frame_num).

Writes one JPEG per stroke to data/videos/frames/{match_id}/set{N}/rally{R}/{frame_num}.jpg
-- the layout already used for the first hand-extracted rally, kept
consistent here so nothing downstream needs two conventions.
"""
from pathlib import Path

import cv2

from shared.models import Rally

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FRAMES_DIR = REPO_ROOT / "data" / "videos" / "frames"


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        return float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()


def check_fps_alignment(video_path: Path, annotation_fps: float, tolerance: float = 0.5) -> float:
    """Refuse a video whose fps differs from ShuttleSet's annotation fps:
    every frame_num would silently point at the wrong moment."""
    video_fps = get_video_fps(video_path)
    if abs(video_fps - annotation_fps) > tolerance:
        raise ValueError(
            f"{video_path.name} is {video_fps:.2f}fps but ShuttleSet annotated this "
            f"match at {annotation_fps:.2f}fps -- frame_num would point at the wrong frames"
        )
    return video_fps


def extract_rally_frames(video_path: Path, rally: Rally, frames_dir: Path = FRAMES_DIR) -> dict[int, Path]:
    """Extract one frame per stroke in `rally`, keyed by frame_num. Idempotent."""
    out_dir = frames_dir / rally.match_id / f"set{rally.set_num}" / f"rally{rally.rally_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    extracted: dict[int, Path] = {}
    try:
        for stroke in rally.strokes:
            dest = out_dir / f"{stroke.frame_num}.jpg"
            if dest.exists():
                extracted[stroke.frame_num] = dest
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, stroke.frame_num)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"could not read frame {stroke.frame_num} from {video_path}")
            cv2.imwrite(str(dest), frame)
            extracted[stroke.frame_num] = dest
    finally:
        cap.release()

    return extracted
