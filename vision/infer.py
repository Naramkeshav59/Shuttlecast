"""Recover a stroke sequence from any badminton broadcast clip -- an upload
or a YouTube link -- so the agent can analyze footage ShuttleSet never
annotated.

    video -> court-view segments (skip replays/close-ups/breaks)
          -> frame features -> hit detector -> hits grouped into rallies
          -> shot classifier per hit -> Rally objects for the agent

Player labels: there's no tracking of who is who, so within each rally the
first hitter (the server) is "A" and hitters alternate -- badminton singles
strokes always alternate between the two players.
"""
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.models import Rally, Stroke  # noqa: E402
from vision.court import GREEN_HI, GREEN_LO, crop, stable_court_box  # noqa: E402
from vision.features import FrameEncoder  # noqa: E402
from vision.train import stroke_context  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "stroke_models.pt"
TRAIN_FPS = 30.0
RALLY_GAP_SEC = 2.5      # a pause longer than this ends a rally
MIN_SEGMENT_SEC = 2.0    # shorter court-view stretches are cutaways, not play
COURT_VIEW_RATIO = 0.18  # share of green pixels that means "main court view"
MIN_RALLY_HITS = 3
MIN_MEDIAN_GAP, MAX_MEDIAN_GAP = 0.4, 2.0  # seconds between strokes in live play


@dataclass
class DetectedRally:
    rally: Rally
    start_sec: float
    shot_confidence: list[float]


def _green_ratio(frame: np.ndarray) -> float:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), GREEN_LO, GREEN_HI)
    return float(mask.mean() / 255.0)


def _looks_like_play(hits: list[int], fps: float) -> bool:
    """Live rallies have a steady rhythm: at least 3 strokes, typically 0.4-2s
    apart. Slow-motion replays shot from the main camera and between-rally
    footage (walking back, picking up the shuttle) fire the detector too, but
    not with that cadence. On raw footage of the held-out match this filter
    is what separates rallies from replays; see vision/eval_raw_video.py."""
    if len(hits) < MIN_RALLY_HITS:
        return False
    gaps = np.diff(hits) / fps
    return bool(MIN_MEDIAN_GAP <= np.median(gaps) <= MAX_MEDIAN_GAP)


def _segments(flags: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """[start, end) runs of True at least min_len long."""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


class StrokeRecognizer:
    def __init__(self):
        import torch

        from vision.train import HitDetector, ShotClassifier, peaks

        self.torch, self.peaks = torch, peaks
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        self.ckpt = ckpt
        # same feature type the models were trained on, recorded in the checkpoint
        self.encoder = FrameEncoder(mode=ckpt.get("features", "pooled"))
        dev = self.encoder.device
        self.detector = HitDetector(ckpt["dim"]).to(dev).eval()
        self.detector.load_state_dict(ckpt["detector"])
        self.classifier = ShotClassifier(ckpt["dim"]).to(dev).eval()
        self.classifier.load_state_dict(ckpt["classifier"])

    def _encode_range(self, video: Path, start: int, end: int, stride: int, box, batch: int = 256) -> np.ndarray:
        """pass 2: decode [start, end) once, encoding every stride-th frame in batches."""
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        feats, buf = [], []
        for i in range(start, end):
            ok, f = cap.read()
            if not ok:
                break
            if (i - start) % stride == 0:
                buf.append(crop(f, box))
            if len(buf) == batch:
                feats.append(self.encoder.encode(buf))
                buf = []
        if buf:
            feats.append(self.encoder.encode(buf))
        cap.release()
        return np.concatenate(feats) if feats else np.zeros((0, self.ckpt["dim"]), np.float32)

    def _court_box(self, video: Path, segments: list[tuple[int, int]], stride: int, n: int = 30):
        """Estimate the crop only from frames INSIDE court-view stretches.
        Sampling at fixed intervals picked up wide shots and graphics with
        green in them, which inflated the box (x 22-1191 vs 158-1123 in
        training) -- a different crop than the models learned from, and
        zero detections on a short clip."""
        picks = []
        total = sum(e - s for s, e in segments)
        for s, e in segments:  # spread n samples across segments by length
            k = max(1, round(n * (e - s) / total))
            picks += [int(s + (e - s) * (i + 0.5) / k) for i in range(k)]
        cap = cv2.VideoCapture(str(video))
        frames = []
        for f in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f * stride)
            ok, img = cap.read()
            if ok:
                frames.append(img)
        cap.release()
        return stable_court_box(frames)

    def run(self, video: Path, match_id: str | None = None, max_minutes: float | None = None,
            progress=None) -> tuple[list[DetectedRally], float]:
        torch, ck = self.torch, self.ckpt
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or TRAIN_FPS
        stride = max(1, round(fps / TRAIN_FPS))   # 50/60fps footage -> ~25/30fps, as trained
        eff_fps = fps / stride
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = int(max_minutes * 60 * fps) if max_minutes else total

        # pass 1: just measure "is the main court on screen" per sampled frame.
        # Holding decoded frames would need ~12 GB for 10 minutes of 720p.
        ratios, idx = [], 0
        while idx < limit:
            ok, f = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                r = _green_ratio(f)
                ratios.append(r)
                if progress and len(ratios) % 500 == 0:
                    progress(f"scanned {len(ratios) / eff_fps / 60:.1f} min of video", min(0.3, 0.3 * idx / max(1, limit)))
            idx += 1
        cap.release()
        court_view = np.convolve(np.array(ratios) > COURT_VIEW_RATIO, np.ones(9) / 9, mode="same") > 0.5
        segments = _segments(court_view, int(MIN_SEGMENT_SEC * eff_fps))
        box = self._court_box(video, segments, stride)
        if not segments or box is None:
            return [], fps

        match_id = match_id or "upload-" + hashlib.sha1(str(video).encode()).hexdigest()[:8]
        results, rally_id = [], 0
        for k, (s, e) in enumerate(segments):
            if progress:
                progress(f"detecting strokes in court segment {k + 1}/{len(segments)}", 0.4 + 0.5 * k / len(segments))
            feats = self._encode_range(video, s * stride, e * stride, stride, box)
            x = ((feats - ck["mean"]) / ck["std"]).astype(np.float32)  # models add motion themselves
            xt = torch.from_numpy(x).to(self.encoder.device)
            with torch.no_grad():
                prob = torch.sigmoid(self.detector(xt[None]))[0].cpu().numpy()
            hits = self.peaks(prob, ck["threshold"], int(0.3 * eff_fps))
            if len(hits) < 2:
                continue
            # split into rallies wherever play pauses
            groups, cur = [], [hits[0]]
            for h in hits[1:]:
                if (h - cur[-1]) / eff_fps > RALLY_GAP_SEC:
                    groups.append(cur)
                    cur = []
                cur.append(h)
            groups.append(cur)
            win = ck["win"]
            padded = np.pad(x, ((win, win), (0, 0)), mode="edge")
            for g in groups:
                if not _looks_like_play(g, eff_fps):
                    continue
                rally_id += 1
                windows = torch.from_numpy(np.stack([padded[h:h + 2 * win + 1] for h in g])).to(self.encoder.device)
                ctx = torch.tensor([stroke_context(np.array(g), i, eff_fps) for i in range(len(g))],
                                   dtype=torch.float32, device=self.encoder.device)
                with torch.no_grad():
                    _, type_logits = self.classifier(windows, ctx)
                    probs = torch.softmax(type_logits, 1).cpu().numpy()
                strokes = [
                    Stroke(
                        rally_id=rally_id, player="A" if i % 2 == 0 else "B", ball_round=i + 1,
                        # frame_num in the ORIGINAL video's frame numbering
                        frame_num=int((s + h) * stride), shot_type=ck["types"][int(p.argmax())],
                        hit_x=None, hit_y=None, landing_x=None, landing_y=None,
                        player_location_x=None, player_location_y=None,
                        opponent_location_x=None, opponent_location_y=None,
                    )
                    for i, (h, p) in enumerate(zip(g, probs))
                ]
                results.append(DetectedRally(
                    rally=Rally(match_id=match_id, set_num=1, rally_id=rally_id, strokes=strokes),
                    start_sec=(s + g[0]) / eff_fps,
                    shot_confidence=[float(p.max()) for p in probs],
                ))
        return results, fps
