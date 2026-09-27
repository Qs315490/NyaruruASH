"""Key EVENTS instead of key STATE - the one reformulation the evidence actually fits.

Everything tried so far asks the same ill-matched question: given what changed between two
frames, say which keys are being HELD.  The label is a state, the evidence is a change.  The
probe series showed the cost: the global motion direction is readable (0.637 for a binary
left/right) while the nine-key state sits at ~0.2 macro-F1.

This asks the matched question instead - given what changed between two frames, say which keys
CHANGED at this tick - so "the picture changed" answers "which key changed".  Same features, same
samples, same linear probe, same metric as probe_key_information.py, so the two numbers are
directly comparable: if events are noticeably more readable than states, the formulation was part
of the wall.

Caveat worth stating: the labels are sampled at 4 fps, so a press and release that both happen
between two samples leaves no event in the labels at all, even though the picture did change.

    uv run python scripts/probe_key_events.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

STEMS = ["BV13HnzzPEEN", "BV19s4y1y7un"]
CAP = 2500


def features_and_labels(stem: str, size: int = 32, cap: int = CAP):
    fr = np.load("data/corpus/%s.npy" % stem, mmap_mode="r")
    L = np.load("runs/keycast-labels-%s.npz" % stem, allow_pickle=True)
    held, names = L["held"].astype(bool), [str(n) for n in L["names"]]
    n = min(len(fr) - 1, len(held) - 1)
    idx = np.arange(1, n)                      # need t-1 for the event, and t+1 for the diff
    if len(idx) > cap:
        idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
    s = 256 // size
    X = np.empty((len(idx), size * size), np.float32)
    for i in range(0, len(idx), 64):
        c = idx[i:i + 64]
        a = np.asarray(fr[c], np.float32)
        b = np.asarray(fr[c + 1], np.float32)
        d = np.abs(a - b).mean(axis=3)
        X[i:i + len(c)] = d.reshape(len(c), size, s, size, s).mean(axis=(2, 4)).reshape(len(c), -1)
        del a, b, d
    event = (held[idx] != held[idx - 1]).astype(np.float32)   # changed at this tick
    state = held[idx].astype(np.float32)                      # held at this tick
    return X, event, state, names


def score(X, Y, cut):
    f1s, P, keep = [], np.zeros((len(Y) - cut, Y.shape[1]), np.float32), []
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
        keep.append(j)
    Yt = Y[cut:][:, keep]
    ones = Yt > 0.5
    onbit = float((P[:, keep][ones] == Yt[ones]).mean()) if ones.any() else float("nan")
    return (float(np.mean(f1s)) if f1s else 0.0), onbit, len(keep), float(Y.mean())


def main() -> int:
    print("%-14s %-8s %9s %9s %9s %8s" % ("video", "target", "macroF1", "ON-bit", "pos rate", "keys"))
    for st in STEMS:
        X, event, state, names = features_and_labels(st)
        cut = int(len(X) * 0.7)
        for tag, Y in (("EVENT", event), ("state", state)):
            f1, onbit, nk, pos = score(X, Y, cut)
            print("%-14s %-8s %9.3f %9.3f %8.1f%% %8d" % (st, tag, f1, onbit, 100 * pos, nk))
    print("\nSame features, same samples, same linear probe - only the question differs.")
    print("EVENT = which key changed at this tick; state = which key is held (what every IDM run")
    print("has asked so far).  A large gap means the formulation, not the representation, was the")
    print("obstacle; no gap means the evidence really only says 'something moved'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
