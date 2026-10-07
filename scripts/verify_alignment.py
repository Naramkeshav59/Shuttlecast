"""Contact sheet of stroke frames spread across a match, for a human to
confirm a downloaded clip actually lines up with ShuttleSet's frame_num.

Matching fps is necessary but not sufficient: a YouTube re-upload with a
different intro shifts every frame. If the sheet shows court views
mid-rally, mark the match aligned in data/verified_videos.json.

    python3 scripts/verify_alignment.py <match_id> <video.mp4> [out.jpg]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.pipeline.frame_extractor import get_video_fps  # noqa: E402
from worker.pipeline.ingest import expected_fps, load_match  # noqa: E402

N_TILES = 8


def main() -> None:
    match_id, video = sys.argv[1], Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"{match_id}_alignment.jpg")

    print(f"video fps {get_video_fps(video):.2f} vs annotated {expected_fps(match_id):.2f}")
    rallies = load_match(match_id)
    picks = np.linspace(0, len(rallies) - 1, N_TILES).astype(int)

    cap = cv2.VideoCapture(str(video))
    tiles = []
    for i in picks:
        stroke = rallies[i].strokes[len(rallies[i].strokes) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, stroke.frame_num)
        ok, frame = cap.read()
        frame = cv2.resize(frame, (320, 180)) if ok else np.zeros((180, 320, 3), np.uint8)
        cv2.putText(frame, f"set{rallies[i].set_num} rally{rallies[i].rally_id}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(frame)
    cap.release()

    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out} -- every tile should be a court view mid-rally")


if __name__ == "__main__":
    main()
