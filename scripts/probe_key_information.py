"""How much key information is in these pixels at all?  A within-video linear probe.

The cross-video IDM fails everywhere, and the failure looks the same whether the backbone is a
small CNN, frozen DINOv2, or a fine-tuned one - which smells like a ceiling rather than a bug.
This measures the ceiling cheaply: per key, a logistic regression on the 64x64 difference image
|a-b| (the interface AGENTS.md §5 measured at 85.7% for a binary move/no-move task), trained and
tested WITHIN one video with a temporal split.  If a linear model reaches only ~0.2 macro-F1,
then no head or backbone is going to make this task easy from single frame pairs.

Reported per video: macro-F1 over keys, accuracy on the bits that are ON in the truth (an
always-empty prediction scores 0), and the video's noop share for context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

STEMS = ["BV19s4y1y7un", "BV13HnzzPEEN", "BV1hc411M7GW"]


def features(stem: str, size: int = 64, cap: int = 3000):
    fr = np.load("data/corpus/%s.npy" % stem, mmap_mode="r")
    L = np.load("runs/keycast-labels-%s.npz" % stem, allow_pickle=True)
    held, names = L["held"].astype(bool), [str(n) for n in L["names"]]
    n = min(len(fr) - 1, len(held) - 1)
    idx = np.arange(n)
    if len(idx) > cap:
        idx = idx[np.linspace(0, n - 1, cap).astype(int)]
    s = 256 // size
    X = np.empty((len(idx), size * size), np.float32)
    for i in range(0, len(idx), 32):                     # chunked: 64-bit floats over 3000
        c = idx[i:i + 32]                                # 256x256 frames would be several GB
        a = np.asarray(fr[c], np.float32)
        b = np.asarray(fr[c + 1], np.float32)
        d = np.abs(a - b).mean(axis=3)
        X[i:i + len(c)] = d.reshape(len(c), size, s, size, s).mean(axis=(2, 4)).reshape(len(c), -1)
        del a, b, d
    return X, held[idx].astype(np.float32), names, float((~held[:n].any(axis=1)).mean())


def main() -> int:
    print("%-14s %7s %9s %9s %10s  %s" % ("video", "samples", "macroF1", "ON-bit", "noop share", "keys"))
    for st in STEMS:
        X, Y, names, noop_share = features(st)
        cut = int(len(Y) * 0.7)
        f1s, P, evaluated = [], np.zeros((len(Y) - cut, Y.shape[1]), np.float32), []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            if y[:cut].sum() < 10 or y[cut:].sum() < 5:
                continue                                     # too rare to score honestly
            clf = LogisticRegression(max_iter=200, C=0.05).fit(X[:cut], y[:cut])
            P[:, j] = clf.decision_function(X[cut:]) > 0
            tp = float((P[:, j] * y[cut:]).sum())
            fp = float((P[:, j] * (1 - y[cut:])).sum())
            fn = float(((1 - P[:, j]) * y[cut:]).sum())
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
            evaluated.append(names[j])
        Yt = Y[cut:]
        ones = Yt > 0.5
        onbit = float((P[ones] == Yt[ones]).mean()) if ones.any() else float("nan")
        print("%-14s %7d %9.3f %9.3f %9.1f%%  %s"
              % (st, len(Y), float(np.mean(f1s)) if f1s else 0.0, onbit, 100 * noop_share,
                 ",".join(evaluated)))
    print("\nRead this as an upper bound on what a LINEAR head can do within one video: if it is")
    print("~0.2 macro-F1, the ceiling itself is low and cross-video failure needs no further")
    print("explanation than that - and no backbone swap will change it much.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
