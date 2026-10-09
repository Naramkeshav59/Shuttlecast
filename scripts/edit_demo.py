"""Cut the raw recording into the demo: each scene at its own speed
(waits fast-forwarded, interactions at 1x), concatenated to an H.264 mp4.

    python scripts/edit_demo.py <out_dir_from_record_demo> <demo.mp4>
"""
import json
import subprocess
import sys
from pathlib import Path

OUT, DEST = Path(sys.argv[1]), sys.argv[2]
meta = json.loads((OUT / "scenes.json").read_text())
scenes = meta["scenes"]

parts = []
for i, (cur, nxt) in enumerate(zip(scenes, scenes[1:])):
    start, end, speed = cur["t"], nxt["t"], cur["speed"]
    parts.append(f"[0:v]trim=start={start:.2f}:end={end:.2f},setpts=(PTS-STARTPTS)/{speed}[v{i}]")
chain = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(parts))) + \
    f"concat=n={len(parts)}:v=1:a=0,fps=30,scale=1280:-2[out]"

subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", meta["video"], "-filter_complex", chain,
                "-map", "[out]", "-c:v", "libx264", "-crf", "23", "-preset", "slow",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", DEST], check=True)
print(DEST)
