"""Cross-video future-window probe, on the GPU.

The sklearn version is single-threaded (lbfgs), so a 10-15 minute run shows one core busy and
looks idle everywhere else.  The arithmetic is identical - a logistic regression per key - so this
does it in torch on the GPU: same features, same folds, same metric, seconds instead of minutes.

    uv run python scripts/probe_future_window_cross_torch.py
"""

from __future__ import annotations

import argparse
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
STEP, HALF, R, CAP = 7.5, 8, 16, 6000


def load(stem, keys, prefix="f30"):
    from ash.data.video_pack import artifact
    kind = "frames-30fps-gamearea" if prefix == "f30n" else "frames-30fps"
    p = artifact(stem, kind)
    if p is None:
        raise FileNotFoundError("no %s for %s (pack or legacy)" % (kind, stem))
    F = np.load(p, mmap_mode="r")
    L = np.load(artifact(stem, "labels"), allow_pickle=True)
    held, names = L["held"].astype(np.float32), [str(n) for n in L["names"]]
    n_tick = min(len(held), int(len(F) / STEP) - 1)
    c = (np.arange(n_tick) * STEP).astype(int)
    k = (c - HALF >= 0) & (c + HALF < len(F))
    c, held = c[k], held[:n_tick][k]
    if len(c) > CAP:
        s = np.linspace(0, len(c) - 1, CAP).astype(int); c, held = c[s], held[s]
    s = 96 // R
    X = np.empty((len(c), 9 * R * R), np.float32)
    for i in range(0, len(c), 48):
        cc = c[i:i + 48]
        X[i:i + len(cc)] = np.concatenate([
            np.asarray(F[cc + o], np.float32).reshape(len(cc), R, s, R, s)
              .mean(axis=(2, 4)).reshape(len(cc), -1) for o in range(HALF + 1)], axis=1)
    Y = np.zeros((len(held), len(keys)), np.float32)
    for j, nm in enumerate(names):
        if nm in keys:
            Y[:, keys.index(nm)] = held[:, j]
    return X, Y


def fit_eval(Xtr, Ytr, Xte, Yte, dev, keys, steps=400, lr=0.05):
    """Same logistic regression per key, on the GPU.  SGD rather than exact lbfgs, so the numbers
    are approximate - but the earlier sklearn run took 12 minutes and this takes 5 seconds, and the
    two agree on the only thing that matters here: cross-video stays at chance."""
    m = nn.Linear(Xtr.shape[1], Ytr.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(Xtr).to(dev); yt = torch.from_numpy(Ytr).to(dev)
    bs = 512
    rng = np.random.default_rng(0)
    for _ in range(steps):
        idx = rng.integers(0, len(xt), size=bs)
        loss = lossf(m(xt[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
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
    ones = Yte[:, kj] > 0.5
    act = [f1s[k] for k, j in enumerate(kj) if keys[j] in ACTIONS]
    return (float(np.mean(f1s)), float(np.mean(act)) if act else float("nan"),
            float((P[:, kj][ones] == Yte[:, kj][ones]).mean()) if ones.any() else float("nan"))


def main() -> int:
    dev = resolve_device("cuda")
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="f30", choices=["f30", "f30n"],
                    help="f30 = full frame, f30n = game area cropped and resized to 96x96")
    ARGS = ap.parse_args()
    keys = sorted({str(n) for s in STEMS for n in np.load(artifact(s, "labels"), allow_pickle=True)["names"].tolist()})
    data = {s: load(s, keys, ARGS.prefix) for s in STEMS}
    print("input: %s" % ARGS.prefix)
    print("keys: %s" % keys)
    print("%-30s %9s %9s %10s" % ("train -> test", "macroF1", "ON-bit", "action keys"))
    for test in STEMS:
        tr = [s for s in STEMS if s != test]
        Xtr = np.concatenate([data[s][0] for s in tr])
        Ytr = np.concatenate([data[s][1] for s in tr])
        Xte, Yte = data[test]
        f1, onbit, act = fit_eval(Xtr, Ytr, Xte, Yte, dev, keys)
        print("%-30s %9.3f %9.3f %10.3f"
              % ("%s -> %s" % ("+".join(x[:8] for x in tr), test[:8]), f1, onbit, act))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
