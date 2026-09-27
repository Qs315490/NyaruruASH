"""Does a longer history window (and more resolution) carry more key information?

The single thing every IDM configuration in this project shares is its input: ONE frame pair.
The within-video linear probe put a ceiling of ~0.23 macro-F1 on that (docs/status.md 二之二十八),
and the direction of the fix is to give the model more of what it lacks - more time and more
pixels - rather than another backbone.

This measures before building anything: per key, a logistic regression on the concatenated
difference images of the last K ticks, downsampled to R x R.  Sweeping K and R separates the two
suspects cheaply:

  * if macro-F1 rises with K, the effect of a press unfolds over several ticks and one pair is
    simply not enough context;
  * if it rises with R, the detail was being averaged away by the downsampling.

Same protocol as probe_key_information.py: temporal split inside one video, macro-F1 over keys,
accuracy on the bits that are ON in the truth (an all-empty prediction scores 0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

KS = [1, 2, 4, 8]          # difference images stacked (K=1 is the single pair used so far)
RS = [32, 64]              # block size the difference image is reduced to
CAP = 2000                 # samples per video


def features(stem: str, K: int, R: int, cap: int = CAP):
    fr = np.load("data/corpus/%s.npy" % stem, mmap_mode="r")
    L = np.load("runs/keycast-labels-%s.npz" % stem, allow_pickle=True)
    held, names = L["held"].astype(bool), [str(n) for n in L["names"]]
    n = min(len(fr) - 1, len(held) - 1)
    lo = K                                       # K diffs need frames t..t-K, so ticks [K, n)
    idx = np.arange(lo, n)
    if len(idx) > cap:
        idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
    s = 256 // R
    X = np.empty((len(idx), K * R * R), np.float32)
    for i in range(0, len(idx), 16):
        c = idx[i:i + 16]
        cols = []
        for j in range(K):                       # j-th most recent difference
            a = np.asarray(fr[c - j], np.float32)
            b = np.asarray(fr[c - j - 1], np.float32)
            d = np.abs(a - b).mean(axis=3)
            cols.append(d.reshape(len(c), R, s, R, s).mean(axis=(2, 4)).reshape(len(c), -1))
            del a, b, d
        X[i:i + len(c)] = np.concatenate(cols, axis=1)
        del cols
    return X, held[idx].astype(np.float32), names


def probe(X, Y, cut):
    f1s, P, scored = [], np.zeros((len(Y) - cut, Y.shape[1]), np.float32), []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if y[:cut].sum() < 10 or y[cut:].sum() < 5:
            continue
        clf = LogisticRegression(max_iter=200, C=0.05).fit(X[:cut], y[:cut])
        P[:, j] = clf.decision_function(X[cut:]) > 0
        tp = float((P[:, j] * y[cut:]).sum())
        fp = float((P[:, j] * (1 - y[cut:])).sum())
        fn = float(((1 - P[:, j]) * y[cut:]).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        scored.append(j)
    Yt = Y[cut:][:, scored]
    ones = Yt > 0.5
    onbit = float((P[:, scored][ones] == Yt[ones]).mean()) if ones.any() else float("nan")
    return float(np.mean(f1s)) if f1s else 0.0, onbit, len(f1s)


def main() -> int:
    stems = sys.argv[1:] or ["BV13HnzzPEEN", "BV19s4y1y7un"]
    for st in stems:
        print("\n=== %s ===" % st)
        print("%4s %4s %10s %9s %6s" % ("K", "R", "macroF1", "ON-bit", "dims"))
        for K in KS:
            for R in RS:
                X, Y, _ = features(st, K, R)
                cut = int(len(Y) * 0.7)
                f1, onbit, nk = probe(X, Y, cut)
                print("%4d %4d %10.3f %9.3f %6d   (%d keys, %d samples)"
                      % (K, R, f1, onbit, X.shape[1], nk, len(Y)))
                del X
    print("\nK = how many consecutive difference images are stacked (K=1 is the single pair every")
    print("IDM run has used so far).  Rising with K means the press's effect needs context;")
    print("rising with R means the downsampling was throwing the detail away.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
