"""Cross-video validation of the future-window recipe - the gate that decides everything.

Within one video, a future window lifts the action keys from 0.11 to 0.68 and a 32x32 grid to
0.86, against 0.19/0.11 for a causal single pair.  But every number measured in this project so
far that mattered was a CROSS-video number, and cross-video has failed for every causal
configuration tried.  This is the same leave-one-video-out protocol that produced those failures,
with the only change being the input: current + 8 future frames at 30 fps, grayscale.

Compromise for cost: a 16x16 grid (2304 dims) instead of the 32x32 best, and 4000 samples per
video, so three folds of eleven keys stay affordable.  The 16x16 within-video reference is 0.594
macro-F1 / 0.681 action keys, so the comparison is directly readable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.data.video_pack import artifact  # noqa: E402

STEMS = ["BV19s4y1y7un", "BV13HnzzPEEN", "BV1hc411M7GW"]
ACTIONS = ["jump", "attack", "dash", "special", "ult"]
STEP = 7.5          # frames per 4 fps tick at 30 fps
HALF = 8            # future frames
R = 16
CAP = 4000


def load(stem: str):
    F = np.load(artifact(stem, "frames-30fps"), mmap_mode="r")
    L = np.load(artifact(stem, "labels"), allow_pickle=True)
    held, names = L["held"].astype(np.float32), [str(n) for n in L["names"]]
    n_tick = min(len(held), int(len(F) / STEP) - 1)
    centres = (np.arange(n_tick) * STEP).astype(int)
    keep = (centres - HALF >= 0) & (centres + HALF < len(F))
    centres, held = centres[keep], held[:n_tick][keep]
    if len(centres) > CAP:
        sel = np.linspace(0, len(centres) - 1, CAP).astype(int)
        centres, held = centres[sel], held[sel]
    s = 96 // R
    X = np.empty((len(centres), 9 * R * R), np.float32)
    for i in range(0, len(centres), 24):
        c = centres[i:i + 24]
        cols = []
        for o in range(0, HALF + 1):
            blk = np.asarray(F[c + o], np.float32)
            cols.append(blk.reshape(len(c), R, s, R, s).mean(axis=(2, 4)).reshape(len(c), -1))
        X[i:i + len(c)] = np.concatenate(cols, axis=1)
    return X, held, names


def wide(held, names, keys):
    Y = np.zeros((len(held), len(keys)), np.float32)
    for j, nm in enumerate(names):
        if nm in keys:
            Y[:, keys.index(nm)] = held[:, j]
    return Y


def main() -> int:
    keys = sorted({n for s in STEMS for n in np.load(artifact(s, "labels"), allow_pickle=True)["names"].tolist()})
    print("keys: %s" % keys)
    data = {}
    for st in STEMS:
        X, held, names = load(st)
        data[st] = (X, wide(held, names, keys))
        print("%-14s %d samples, %d keys" % (st, len(X), len(names)))

    print("\n%-34s %9s %9s %10s" % ("train -> test", "macroF1", "ON-bit", "action keys"))
    for test in STEMS:
        train = [s for s in STEMS if s != test]
        Xtr = np.concatenate([data[s][0] for s in train])
        Ytr = np.concatenate([data[s][1] for s in train])
        Xte, Yte = data[test]
        f1s, P, kj = [], np.zeros((len(Yte), len(keys)), np.float32), []
        for j in range(len(keys)):
            y = Ytr[:, j]
            if y.sum() < 20 or Yte[:, j].sum() < 5:
                continue
            clf = LogisticRegression(max_iter=200, C=0.05).fit(Xtr, y)
            P[:, j] = clf.decision_function(Xte) > 0
            tp = float((P[:, j] * Yte[:, j]).sum())
            fp = float((P[:, j] * (1 - Yte[:, j])).sum())
            fn = float(((1 - P[:, j]) * Yte[:, j]).sum())
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
            kj.append(j)
        ones = Yte[:, kj] > 0.5
        act = [f1s[k] for k, j in enumerate(kj) if keys[j] in ACTIONS]
        print("%-34s %9.3f %9.3f %10.3f"
              % ("%s -> %s" % ("+".join(s[:8] for s in train), test[:8]),
                 float(np.mean(f1s)),
                 float((P[:, kj][ones] == Yte[:, kj][ones]).mean()) if ones.any() else float("nan"),
                 float(np.mean(act)) if act else float("nan")))
    print("\nWithin-video reference (same 16x16 grid, future 8): macro-F1 0.594, action keys 0.681.")
    print("Causal single pair, 4 fps, for contrast:                  macro-F1 0.197, action keys 0.108.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
