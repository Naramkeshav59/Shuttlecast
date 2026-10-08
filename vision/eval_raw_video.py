"""End-to-end check of the upload pipeline on RAW footage of the held-out
test match: replays, close-ups and breaks included, exactly as a user's
upload would be. Detected strokes are matched against ShuttleSet's
ground truth for the same stretch of video.

The training eval scores pre-cut rally windows; this one also exercises
court-view segmentation and rally grouping, so it's the honest number for
"analyze any video".

    python vision/eval_raw_video.py [start_min] [minutes]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vision.infer import StrokeRecognizer  # noqa: E402
from vision.train import HIT_TOL_SEC, SHOT_FAMILY, TEST_MATCHES  # noqa: E402
from worker.pipeline.ingest import load_match, verified_videos  # noqa: E402


def main() -> None:
    start_min = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    match_id = next(iter(TEST_MATCHES))
    video = verified_videos()[match_id]

    import cv2
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
    lo, hi = int(start_min * 60 * fps), int((start_min + minutes) * 60 * fps)

    # cut the stretch to a temp clip so the pipeline sees it like an upload
    import subprocess
    clip = REPO_ROOT / "data" / "uploads" / "eval_clip.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start_min * 60), "-t", str(minutes * 60),
                    "-i", str(video), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(clip)], check=True)

    detected, clip_fps = StrokeRecognizer().run(clip)
    det = [(s.frame_num / clip_fps, s.shot_type) for d in detected for s in d.rally.strokes]
    truth = [((s.frame_num - lo) / fps, s.shot_type) for r in load_match(match_id)
             for s in r.strokes if lo <= s.frame_num < hi]

    used, tp, fam_ok, fam_n = set(), 0, 0, 0
    for t, shot in truth:
        cands = [i for i, (d, _) in enumerate(det) if abs(d - t) <= HIT_TOL_SEC and i not in used]
        if cands:
            i = min(cands, key=lambda j: abs(det[j][0] - t))
            used.add(i)
            tp += 1
            if shot in SHOT_FAMILY:
                fam_n += 1
                fam_ok += SHOT_FAMILY.get(det[i][1]) == SHOT_FAMILY[shot]
    prec = tp / len(det) if det else 0.0
    rec = tp / len(truth) if truth else 0.0
    report = {
        "clip": f"{match_id} {start_min}-{start_min + minutes} min (raw footage)",
        "true_strokes": len(truth), "detected_strokes": len(det),
        "detected_rallies": len(detected),
        "true_rallies": len({r.rally_id for r in load_match(match_id) for s in r.strokes if lo <= s.frame_num < hi}),
        "precision": round(prec, 3), "recall": round(rec, 3),
        "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        "shot_family_accuracy_on_matched": round(fam_ok / fam_n, 3) if fam_n else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
