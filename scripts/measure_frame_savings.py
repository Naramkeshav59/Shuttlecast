"""Measure the frame-count saving of stroke-timestamp extraction vs uniform
sampling, on every verified match. Frames sent to the vision model are the
cost driver (tokens per image), so that's what's counted.

Baselines, both at 1 frame/sec:
  * whole video  -- what a pipeline without stroke data has to do
  * rally windows only -- a generous baseline that already knows where the
    rallies are and just samples uniformly inside them

    python3 scripts/measure_frame_savings.py
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.pipeline.ingest import load_match, verified_videos  # noqa: E402

SAMPLE_FPS = 1.0


def main() -> None:
    totals = {"stroke": 0, "rally_windows": 0, "whole_video": 0}
    print(f"{'match':42s} {'stroke':>7s} {'1fps rallies':>13s} {'1fps video':>11s}")
    for match_id, video in verified_videos().items():
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()

        rallies = [r for r in load_match(match_id) if r.strokes]
        stroke = sum(len(r.strokes) for r in rallies)
        windows = sum(
            int((r.strokes[-1].frame_num - r.strokes[0].frame_num) / fps * SAMPLE_FPS) + 1
            for r in rallies
        )
        whole = int(duration * SAMPLE_FPS)
        totals["stroke"] += stroke
        totals["rally_windows"] += windows
        totals["whole_video"] += whole
        print(f"{match_id[:42]:42s} {stroke:7d} {windows:13d} {whole:11d}")

    s, w, v = totals["stroke"], totals["rally_windows"], totals["whole_video"]
    print(f"{'TOTAL':42s} {s:7d} {w:13d} {v:11d}")
    print(f"reduction vs 1fps inside rally windows: {1 - s / w:.1%}")
    print(f"reduction vs 1fps over the whole video: {1 - s / v:.1%}")


if __name__ == "__main__":
    main()
