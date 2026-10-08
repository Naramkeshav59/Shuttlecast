"""Per-frame visual features for stroke detection and shot classification.

Every frame is court-cropped (vision/court.py) and embedded by a frozen,
pretrained DINOv2-small. Only the small temporal heads in vision/train.py
are trained -- a few thousand labelled strokes are enough for those, not
for a video backbone.
"""
from pathlib import Path

import cv2
import numpy as np

from vision.court import crop, stable_court_box

BACKBONE = "facebook/dinov2-small"
GRID = 4  # patch tokens average-pooled to a GRID x GRID map: keeps WHERE players are
# "pooled": CLS + mean patch (768). "grid": CLS + 4x4 pooled patches (384 * 17).
# Grid features exist because shot type depends on court position (net shot vs
# clear), which a single mean-pooled vector throws away.
FEATURE_DIMS = {"pooled": 768, "grid": 384 * (1 + GRID * GRID)}


class FrameEncoder:
    def __init__(self, device: str | None = None, mode: str = "grid"):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(BACKBONE).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def encode(self, crops: list[np.ndarray], batch: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(crops), batch):
            rgb = np.stack([c[:, :, ::-1] for c in crops[i:i + batch]])  # BGR -> RGB
            x = torch.from_numpy(rgb).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - self.mean) / self.std
            if self.device == "cuda":
                x = x.half()
            with torch.no_grad():
                h = self.model(pixel_values=x).last_hidden_state
            cls, patches = h[:, 0], h[:, 1:]
            if self.mode == "grid":
                side = int(patches.shape[1] ** 0.5)  # 16x16 patches at 224px
                grid = patches.reshape(-1, side, side, patches.shape[2]).permute(0, 3, 1, 2)
                grid = torch.nn.functional.adaptive_avg_pool2d(grid, GRID).flatten(1)
                vec = torch.cat([cls, grid], dim=1)
            else:
                vec = torch.cat([cls, patches.mean(1)], dim=1)
            out.append(vec.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, FEATURE_DIMS[self.mode]), np.float32)


def read_frames(video: Path, start: int, end: int) -> list[np.ndarray]:
    """Frames [start, end) read sequentially after one seek (seeking per
    frame would be far slower)."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start))
    frames = []
    for _ in range(max(0, end - start)):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def court_box_for(video: Path, sample_frames: list[int]):
    cap = cv2.VideoCapture(str(video))
    frames = []
    for n in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return stable_court_box(frames)


def embed_window(encoder: FrameEncoder, video: Path, start: int, end: int, box) -> np.ndarray:
    frames = read_frames(video, start, end)
    return encoder.encode([crop(f, box) for f in frames])
