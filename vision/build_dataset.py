"""Turn ShuttleSet annotations + verified match videos into a training set
for the stroke models: per-rally frame features plus labels (which frames
are hits, and each hit's shot type).

One .npz per rally under data/vision/features/{match_id}/. Resumable.

    python vision/build_dataset.py [grid|pooled]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vision.features import FrameEncoder, court_box_for, embed_window  # noqa: E402
from worker.pipeline.frame_extractor import check_fps_alignment  # noqa: E402
from worker.pipeline.ingest import expected_fps, load_match, verified_videos  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "grid"
OUT = REPO_ROOT / "data" / "vision" / ("features_grid" if MODE == "grid" else "features")
PAD_SEC = 1.0  # context before the serve and after the last stroke
GAP_MARGIN_SEC = 1.5  # stay clear of the rally edges when sampling negatives
GAP_MAX_SEC = 10.0


def main() -> None:
    encoder = FrameEncoder(mode=MODE)
    print(f"encoder on {encoder.device}", file=sys.stderr)
    for match_id, video in verified_videos().items():
        fps = check_fps_alignment(video, expected_fps(match_id))
        rallies = [r for r in load_match(match_id) if len(r.strokes) >= 2]
        box = court_box_for(video, [r.strokes[0].frame_num for r in rallies[:: max(1, len(rallies) // 12)]])
        if box is None:
            print(f"skip {match_id}: no court found", file=sys.stderr)
            continue
        out_dir = OUT / match_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(json.dumps({"fps": fps, "court_box": box}), encoding="utf-8")
        t0, n_frames = time.time(), 0
        for r in rallies:
            path = out_dir / f"set{r.set_num}_rally{r.rally_id}.npz"
            if path.exists():
                continue
            start = r.strokes[0].frame_num - int(PAD_SEC * fps)
            end = r.strokes[-1].frame_num + int(PAD_SEC * fps)
            feats = embed_window(encoder, video, start, end, box)
            hits = np.array([s.frame_num - start for s in r.strokes])
            np.savez_compressed(
                path, feats=feats.astype(np.float16), hits=hits,
                shots=np.array([s.shot_type for s in r.strokes]),
                winner=np.array(r.rally_winner or ""), start=start,
            )
            n_frames += len(feats)
        # Hard negatives: court footage BETWEEN rallies (walking back, picking up
        # the shuttle, main-camera slow-mo replays). Without these the detector
        # only ever saw rally footage, and on raw uploads it fired on replays
        # (precision 0.49 vs 0.83 on pre-cut windows).
        n_gaps = 0
        for prev, nxt in zip(rallies, rallies[1:]):
            if prev.set_num != nxt.set_num:
                continue  # between sets: long breaks, often off-court footage
            path = out_dir / f"gap_set{prev.set_num}_after{prev.rally_id}.npz"
            if path.exists():
                continue
            start = prev.strokes[-1].frame_num + int(GAP_MARGIN_SEC * fps)
            end = min(nxt.strokes[0].frame_num - int(GAP_MARGIN_SEC * fps), start + int(GAP_MAX_SEC * fps))
            if end - start < int(2 * fps):
                continue
            feats = embed_window(encoder, video, start, end, box)
            np.savez_compressed(path, feats=feats.astype(np.float16), hits=np.array([], dtype=int),
                                shots=np.array([]), winner=np.array(""), start=start)
            n_gaps += 1
        print(f"{match_id}: {len(rallies)} rallies, {n_frames} frames, {n_gaps} gap negatives "
              f"in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
