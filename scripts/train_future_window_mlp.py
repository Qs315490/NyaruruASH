"""Does augmentation buy cross-video transfer?  An A/B in one script.

Established: with a future window, a linear probe reaches 0.59-0.73 macro-F1 WITHIN one video
(action keys 0.68-0.86) and collapses to 0.006-0.060 ACROSS videos.  The information is there; the
transfer is not.  So the problem is now well posed as domain generalisation, and the standard first
move is augmentation - breaking whatever it is the model keys on instead of the action.

Augmentations, each chosen against a specific capture-pipeline difference:
  * brightness / contrast gain    -> exposure and encoder differences
  * random spatial shift (crop)   -> framing and window-position differences
  * per-video standardisation     -> per-video mean/variance of the pixel distribution
NOT horizontal flip: the direction keys are positional, so flipping would swap left and right and
teach the model something false.

Same folds, same features, same metric, with and without augmentation, so the number is readable
on its own.  Cross-video baseline to beat: macro-F1 0.006 / 0.060 / 0.025 (torch probe).

    uv run python scripts/train_future_window_mlp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.utils.device import resolve_device  # noqa: E402

from ash.data.video_pack import artifact as _artifact  # noqa: E402


def _path(stem: str, kind: str) -> str:
    """Resolve an artifact through the video pack, with its legacy fallback.

    Hard-coding runs/<name> is what broke these scripts when the packs were introduced, so the
    layout is now looked up rather than assumed.
    """
    p = _artifact(stem, kind)
    if p is None:
        raise SystemExit("missing %s for %s" % (kind, stem))
    return str(p)



STEMS = ["BV19s4y1y7un", "BV13HnzzPEEN", "BV1hc411M7GW"]
ACTIONS = ["jump", "attack", "dash", "special", "ult"]
STEP, HALF, R = 7.5, 8, 16
S = 96 // R


class MLP(nn.Module):
    def __init__(self, d_in: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_index(stem: str, keys: list[str]):
    from ash.data.video_pack import artifact
    p = artifact(stem, "frames-30fps")
    if p is None:
        raise FileNotFoundError("no frames-30fps for %s" % stem)
    F = np.load(p, mmap_mode="r")
    L = np.load(artifact(stem, "labels"), allow_pickle=True)
    held, names = L["held"].astype(np.float32), [str(n) for n in L["names"]]
    n_tick = min(len(held), int(len(F) / STEP) - 1)
    c = (np.arange(n_tick) * STEP).astype(int)
    k = (c - HALF - 6 >= 0) & (c + HALF + 6 < len(F))
    c, held = c[k], held[:n_tick][k]
    Y = np.zeros((len(held), len(keys)), np.float32)
    for j, nm in enumerate(names):
        if nm in keys:
            Y[:, keys.index(nm)] = held[:, j]
    return F, c, Y


def featurize(F, c, rng=None, augment=False):
    """(N, 9*R*R) from the 9 frames of each window, optionally augmented first."""
    X = np.empty((len(c), 9 * R * R), np.float32)
    for i in range(0, len(c), 64):
        cc = c[i:i + 64]
        gain = bias = 0.0
        dx = dy = 0
        if augment:
            gain = rng.uniform(0.75, 1.25)
            bias = rng.uniform(-20, 20)
            dx, dy = rng.integers(-6, 7, size=2)
        cols = []
        for o in range(HALF + 1):
            blk = np.asarray(F[cc + o], np.float32)
            if augment and (dx or dy):
                blk = np.roll(blk, (int(dy), int(dx)), axis=(1, 2))
            if augment:
                blk = blk * gain + bias
            cols.append(blk.reshape(len(cc), R, S, R, S).mean(axis=(2, 4)).reshape(len(cc), -1))
        X[i:i + len(cc)] = np.concatenate(cols, axis=1)
    return X


def run(Xtr, Ytr, Xte, Yte, keys, dev, epochs=12, bs=256, lr=1e-3, standardize=True):
    if standardize:                       # per-video mean/std removes a pipeline-level offset
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    m = MLP(Xtr.shape[1], Ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(Xtr).to(dev); yt = torch.from_numpy(Ytr).to(dev)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        m.train()
        for _ in range(max(1, len(xt) // bs)):
            idx = rng.integers(0, len(xt), size=bs)
            loss = lossf(m(xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.inference_mode():
        P = (m(torch.from_numpy(Xte).to(dev)) > 0).float().cpu().numpy()
    f1s, kj = [], []
    for j in range(Yte.shape[1]):
        y = Yte[:, j]
        if y.sum() < 5:
            continue
        tp = float((P[:, j] * y).sum()); fp = float((P[:, j] * (1 - y)).sum())
        fn = float(((1 - P[:, j]) * y).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        kj.append(j)
    act = [f1s[k] for k, j in enumerate(kj) if keys[j] in ACTIONS]
    return float(np.mean(f1s)), float(np.mean(act)) if act else float("nan")


def main() -> int:
    dev = resolve_device("cuda")
    keys = sorted({str(n) for s in STEMS for n in np.load(artifact(s, "labels"), allow_pickle=True)["names"].tolist()})
    idx = {s: load_index(s, keys) for s in STEMS}
    print("%-30s %9s %11s  %s" % ("train -> test", "macroF1", "action keys", "condition"))
    for test in STEMS:
        tr = [s for s in STEMS if s != test]
        for augment in (False, True):
            rng = np.random.default_rng(1)
            Xtr = np.concatenate([featurize(idx[s][0], idx[s][1], rng, augment) for s in tr])
            Ytr = np.concatenate([idx[s][2] for s in tr])
            Xte = featurize(idx[test][0], idx[test][1], None, False)
            f1, act = run(Xtr, Ytr, Xte, idx[test][2], keys, dev)
            print("%-30s %9.3f %11.3f  %s"
                  % ("%s -> %s" % ("+".join(x[:8] for x in tr), test[:8]), f1, act,
                     "augmented" if augment else "plain"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
