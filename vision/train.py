"""Train the two stroke models on ShuttleSet-labelled frame features.

* Hit detector: per-frame hit likelihood over a rally (temporal conv net),
  peak-picked and scored as F1 within +/-0.15s -- the same scoring as the
  audio-onset baseline it replaces.
* Shot classifier: the ~0.5s around each hit plus the rally's timing
  (flight time to the next hit) -> shot family (7 classes, primary
  metric) and shot type (18, secondary).

Both models compute frame-to-frame motion internally, so callers pass only
normalized features -- the same call in training and in vision/infer.py.
Features stay float16 in RAM and are normalized per rally on the GPU:
grid features are 6,528 numbers per frame, too big to expand upfront.

Split by MATCH (the test match is never seen in training).
Writes models/stroke_models.pt and models/stroke_eval.json.

    python vision/train.py [grid|pooled]
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from worker.eval.metrics import SHOT_FAMILY  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else "grid"
MODELS = REPO_ROOT / "models"
TEST_MATCHES = {"Evgeniya_Kosetskaya_Michelle_Li_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals"}
HIT_TOL_SEC = 0.15
MIN_GAP_SEC = 0.3
SIGMA = 1.5          # frames, width of the target bump around each hit
WIN = 8              # frames either side of a hit for the shot classifier
FAMILIES = sorted(set(SHOT_FAMILY.values()))
TYPES = sorted(SHOT_FAMILY)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CTX = 4


def feats_dir(mode: str) -> Path:
    return REPO_ROOT / "data" / "vision" / ("features_grid" if mode == "grid" else "features")


def stroke_context(hits: np.ndarray, i: int, fps: float) -> list[float]:
    """Shared by training and vision/infer.py so both build identical inputs.
    The gap to the next hit is the shuttle's flight time: a smash arrives in
    a fraction of a second, a clear takes over one."""
    prev_gap = (hits[i] - hits[i - 1]) / fps if i > 0 else 0.0
    next_gap = (hits[i + 1] - hits[i]) / fps if i + 1 < len(hits) else 0.0
    return [float(i == 0), float(i == len(hits) - 1), float(min(prev_gap, 3.0)), float(min(next_gap, 3.0))]


def add_motion(x: torch.Tensor) -> torch.Tensor:
    """(B, T, D) -> (B, T, 2D): features plus frame-to-frame change.
    A hit is a change, not a pose."""
    d = torch.diff(x, dim=1, prepend=x[:, :1])
    return torch.cat([x, d], dim=2)


class HitDetector(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.inp = nn.Linear(2 * dim, hidden)
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 8))
        self.out = nn.Conv1d(hidden, 1, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):  # x: (B, T, D) normalized features
        h = self.drop(F.gelu(self.inp(add_motion(x)))).transpose(1, 2)
        for conv in self.convs:
            h = h + F.gelu(conv(h))
        return self.out(h).squeeze(1)  # (B, T) logits


class ShotClassifier(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.inp = nn.Linear(2 * dim + N_CTX, hidden)
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=1)
        # not `self.type`: that would shadow nn.Module.type() and break calls
        self.family_head = nn.Linear(2 * hidden, len(FAMILIES))
        self.type_head = nn.Linear(2 * hidden, len(TYPES))
        self.drop = nn.Dropout(0.3)

    def forward(self, x, ctx):  # x: (B, 2W+1, D) normalized features, ctx: (B, N_CTX)
        x = add_motion(x)
        c = ctx[:, None, :].expand(-1, x.shape[1], -1)
        h = self.drop(F.gelu(self.inp(torch.cat([x, c], dim=2)))).transpose(1, 2)
        h = F.gelu(self.conv(h))
        pooled = torch.cat([h.mean(2), h.amax(2)], dim=1)
        return self.family_head(pooled), self.type_head(pooled)


def load(mode: str) -> dict[str, list[dict]]:
    data = {}
    for match_dir in sorted(p for p in feats_dir(mode).iterdir() if p.is_dir()):
        meta = json.loads((match_dir / "meta.json").read_text(encoding="utf-8"))
        rallies = []
        for f in sorted(match_dir.glob("*.npz")):
            z = np.load(f)
            rallies.append({"feats": z["feats"], "hits": z["hits"],  # float16, kept compact
                            "shots": [str(s) for s in z["shots"]], "fps": meta["fps"]})
        data[match_dir.name] = rallies
    return data


def norm_stats(rallies) -> tuple[np.ndarray, np.ndarray]:
    total = sq = None
    n = 0
    for r in rallies:  # streaming, so the float64 sums never hold every frame
        f = r["feats"].astype(np.float64)
        total = f.sum(0) if total is None else total + f.sum(0)
        sq = (f ** 2).sum(0) if sq is None else sq + (f ** 2).sum(0)
        n += len(f)
    mean = total / n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 0)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.from_numpy(mean).to(DEVICE)
        self.std = torch.from_numpy(std).to(DEVICE)

    def __call__(self, feats: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(feats).to(DEVICE).float() - self.mean) / self.std


def heat_target(n: int, hits: np.ndarray) -> np.ndarray:
    t = np.arange(n)[:, None]
    return np.exp(-0.5 * ((t - hits[None, :]) / SIGMA) ** 2).max(axis=1) if len(hits) else np.zeros(n)


def peaks(prob: np.ndarray, thr: float, min_gap: int) -> np.ndarray:
    idx = [i for i in range(len(prob))
           if prob[i] >= thr and prob[i] == prob[max(0, i - min_gap):i + min_gap + 1].max()]
    keep = []
    for i in idx:  # flat tops can tie; keep the first of each cluster
        if not keep or i - keep[-1] > min_gap:
            keep.append(i)
    return np.array(keep)


def f1_at(rallies, probs, thr):
    tp = fp = fn = 0
    for r, p in zip(rallies, probs):
        tol = HIT_TOL_SEC * r["fps"]
        det = peaks(p, thr, int(MIN_GAP_SEC * r["fps"]))
        used = set()
        for h in r["hits"]:
            cands = [j for j, d in enumerate(det) if abs(d - h) <= tol and j not in used]
            if cands:
                used.add(min(cands, key=lambda j: abs(det[j] - h)))
                tp += 1
        fp += len(det) - len(used)
        fn += len(r["hits"]) - len(used)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def train_detector(train, norm, dim, epochs=40):
    model = HitDetector(dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    pos_weight = torch.tensor(8.0, device=DEVICE)
    for _ in range(epochs):
        model.train()
        random.shuffle(train)
        for r in train:
            x = norm(r["feats"])[None]
            y = torch.from_numpy(heat_target(len(r["feats"]), r["hits"])).float().to(DEVICE)[None]
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def predict(model, rallies, norm):
    model.eval()
    with torch.no_grad():
        return [torch.sigmoid(model(norm(r["feats"])[None]))[0].cpu().numpy() for r in rallies]


def hit_windows(rallies):
    """Raw float16 windows around each labelled hit (normalized later, on GPU)."""
    xs, ctx, fam, typ = [], [], [], []
    for r in rallies:
        f = r["feats"]
        padded = np.pad(f, ((WIN, WIN), (0, 0)), mode="edge")
        for i, (h, shot) in enumerate(zip(r["hits"], r["shots"])):
            if shot not in SHOT_FAMILY or not (0 <= h < len(f)):
                continue  # 'unknown' shots carry no label
            xs.append(padded[h:h + 2 * WIN + 1])
            ctx.append(stroke_context(r["hits"], i, r["fps"]))
            fam.append(FAMILIES.index(SHOT_FAMILY[shot]))
            typ.append(TYPES.index(shot))
    return (np.stack(xs), torch.tensor(ctx, dtype=torch.float32), torch.tensor(fam), torch.tensor(typ))


def train_classifier(w, norm, dim, epochs=60):
    x, ctx, fam, typ = w
    model = ShotClassifier(dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # balance classes: rare shots (rush, driven flight) otherwise get ignored
    fw = torch.tensor([1.0 / max(1, (fam == i).sum().item()) for i in range(len(FAMILIES))], device=DEVICE)
    tw = torch.tensor([1.0 / max(1, (typ == i).sum().item()) for i in range(len(TYPES))], device=DEVICE)
    fw, tw = fw / fw.mean(), tw / tw.mean()
    n = len(x)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            b = perm[i:i + 64].numpy()
            lf, lt = model(norm(x[b]), ctx[b].to(DEVICE))
            loss = (F.cross_entropy(lf, fam[b].to(DEVICE), weight=fw)
                    + 0.5 * F.cross_entropy(lt, typ[b].to(DEVICE), weight=tw))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def classify_eval(model, w, norm):
    x, ctx, fam, typ = w
    model.eval()
    pf, pt = [], []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            lf, lt = model(norm(x[i:i + 128]), ctx[i:i + 128].to(DEVICE))
            pf.append(lf.argmax(1).cpu()); pt.append(lt.argmax(1).cpu())
    pf, pt = torch.cat(pf), torch.cat(pt)
    macro = np.mean([((pf == c) & (fam == c)).sum().item() / max(1, (fam == c).sum().item())
                     for c in range(len(FAMILIES)) if (fam == c).any()])
    return {"family_accuracy": (pf == fam).float().mean().item(), "family_balanced_accuracy": float(macro),
            "type_accuracy": (pt == typ).float().mean().item(), "n": len(fam)}


def main() -> None:
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    data = load(MODE)
    test_ids = [m for m in data if m in TEST_MATCHES]
    rest = [m for m in data if m not in TEST_MATCHES]
    val_ids, train_ids = rest[-1:], rest[:-1]
    print(f"features={MODE}\ntrain {train_ids}\nval {val_ids}\ntest {test_ids}", file=sys.stderr)

    train = [r for m in train_ids for r in data[m]]
    val = [r for m in val_ids for r in data[m]]
    test = [r for m in test_ids for r in data[m]]
    mean, std = norm_stats(train)
    norm = Normalizer(mean, std)
    dim = train[0]["feats"].shape[1]

    detector = train_detector(list(train), norm, dim)
    val_probs = predict(detector, val, norm)
    best_thr = max((round(t, 2) for t in np.arange(0.1, 0.95, 0.05)), key=lambda t: f1_at(val, val_probs, t)["f1"])
    hit_test = f1_at(test, predict(detector, test, norm), best_thr)

    train_w = hit_windows(train)
    clf = train_classifier(train_w, norm, dim)
    test_w = hit_windows(test)
    shot_test = classify_eval(clf, test_w, norm)
    majority = Counter(train_w[2].tolist()).most_common(1)[0][0]
    shot_test["majority_baseline_family_accuracy"] = (test_w[2] == majority).float().mean().item()

    report = {
        "features": MODE,
        "split": {"train": train_ids, "val": val_ids, "test": test_ids},
        "n_train_strokes": int(sum(len(r["hits"]) for r in train)),
        "hit_detection_test": {**hit_test, "threshold": best_thr, "tolerance_sec": HIT_TOL_SEC,
                               "audio_onset_baseline_f1": 0.37},
        "shot_classification_test": shot_test,
    }
    MODELS.mkdir(exist_ok=True)
    torch.save({"detector": detector.state_dict(), "classifier": clf.state_dict(), "mean": mean, "std": std,
                "threshold": best_thr, "families": FAMILIES, "types": TYPES, "win": WIN,
                "dim": dim, "features": MODE}, MODELS / "stroke_models.pt")
    (MODELS / "stroke_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "split"}, indent=2))


if __name__ == "__main__":
    main()
