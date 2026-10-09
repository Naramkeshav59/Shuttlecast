"""Cut raw recordings into the demo: each scene at its own speed (waits
fast-forwarded, interactions at 1x), all parts concatenated to one H.264 mp4.

    python scripts/edit_demo.py demo.mp4 run1:qa run2 [--recaption run1/results=caption.txt]
        run1:qa  -> scenes of run1 up to (not including) its 'qa' scene
        run2     -> all of run2
        --recaption folder/scene=file  covers that scene's burned-in caption
                                       with the text in file (when what
                                       happened on screen differs from the
                                       caption written before the run)
"""
import argparse
import json
import subprocess
from pathlib import Path

FONT = "C\\:/Windows/Fonts/segoeuib.ttf"
# caption box as drawn by record_demo.py at 1440x900: centered, bottom 28px
BOX = "drawbox=x=280:y=760:w=880:h=130:color=0x111827@1:t=fill"

ap = argparse.ArgumentParser()
ap.add_argument("dest")
ap.add_argument("specs", nargs="+")
ap.add_argument("--recaption", action="append", default=[])
args = ap.parse_args()

recaption = {}
for item in args.recaption:
    key, _, textfile = item.partition("=")
    recaption[key] = Path(textfile).resolve().as_posix().replace(":", "\\:")

inputs, parts = [], []
for n_in, spec in enumerate(args.specs):
    folder, _, until = spec.rpartition(":")
    if not folder or "/" in until or "\\" in until:  # no scene suffix, just a (Windows) path
        folder, until = spec, ""
    meta = json.loads((Path(folder) / "scenes.json").read_text())
    scenes = meta["scenes"]
    if until:
        cut = next(i for i, s in enumerate(scenes) if s["name"] == until)
        scenes = scenes[:cut + 1]  # keep the 'until' marker as the end time
    inputs += ["-i", meta["video"]]
    for cur, nxt in zip(scenes, scenes[1:]):
        n = len(parts)
        f = f"[{n_in}:v]trim=start={cur['t']:.2f}:end={nxt['t']:.2f},setpts=(PTS-STARTPTS)/{cur['speed']}"
        textfile = recaption.get(f"{Path(folder).name}/{cur['name']}")
        if textfile:
            f += (f",{BOX},drawtext=fontfile='{FONT}':textfile='{textfile}':fontcolor=white:fontsize=30:"
                  f"line_spacing=8:x=(w-text_w)/2:y=825-text_h/2")
        parts.append(f + f"[v{n}]")

chain = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(parts))) + \
    f"concat=n={len(parts)}:v=1:a=0,fps=30,scale=1280:-2[out]"
subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", chain,
                "-map", "[out]", "-c:v", "libx264", "-crf", "23", "-preset", "slow",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.dest], check=True)
print(args.dest)
