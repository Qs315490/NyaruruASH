"""Is left/right motion even present in the features?  A linear probe answers it.

The fine-tuned IDM calls `right` "left" everywhere (right has 4417 positives in video 2 and is
predicted zero times), while the labels are verified against picture motion.  Before blaming the
training recipe, ask the cheaper question: can a LINEAR classifier on the frozen DINOv2 features
tell left from right at all?

  * if it can, the information is in the features and the IDM's failure is optimisation;
  * if it cannot, no head will fix it and the representation is the wall, as the earlier
    experiments suggested.

The probe reads the feature DIFFERENCE between consecutive frames (E[t+1] - E[t]), because that
is where the direction of motion lives; the same features the IDM saw are used, so the two
experiments are comparable.  Chance level is reported from the test set's own balance.

    uv run python scripts/probe_direction_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.data.video_pack import artifact  # noqa: E402


def samples(stem: str):
    E = np.load(Path("runs") / ("dino-%s.npy" % stem))
    L = np.load(artifact(stem, "labels"), allow_pickle=True)
    held, names = L["held"].astype(bool), [str(n) for n in L["names"]]
    if "left" not in names or "right" not in names:
        return None
    il, ir = names.index("left"), names.index("right")
    n = min(len(E) - 1, len(held) - 1)
    # Only ticks where exactly one of the two is held: a clean left-vs-right question.
    onlyl = held[:n, il] & ~held[:n, ir]
    onlyr = held[:n, ir] & ~held[:n, il]
    idx = np.where(onlyl | onlyr)[0]
    X = (E[idx + 1] - E[idx]).astype(np.float64)
    y = onlyr[idx].astype(int)
    return X, y


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    stems = ["BV19s4y1y7un", "BV13HnzzPEEN", "BV1hc411M7GW"]
    data = {}
    for st in stems:
        s = samples(st)
        if s is None:
            print("%-14s no left/right labels" % st)
            continue
        X, y = s
        data[st] = (X, y)
        print("%-14s %5d ticks with exactly one of left/right  (right %d)" % (st, len(y), y.sum()))

    res = {}
    print("\n-- within video (temporal split: first 70%% train, last 30%% test) --")
    for st, (X, y) in data.items():
        cut = int(len(y) * 0.7)
        if cut < 50 or len(y) - cut < 50:
            print("%-14s too few samples" % st)
            continue
        sc = StandardScaler().fit(X[:cut])
        clf = LogisticRegression(max_iter=1000, C=0.1).fit(sc.transform(X[:cut]), y[:cut])
        acc = float(clf.score(sc.transform(X[cut:]), y[cut:]))
        chance = float(max(y[cut:].mean(), 1 - y[cut:].mean()))
        res["within_%s" % st] = {"acc": round(acc, 3), "chance": round(chance, 3)}
        print("%-14s accuracy %.3f   chance %.3f" % (st, acc, chance))

    print("\n-- across videos (train on one, test on another) --")
    names = list(data)
    for a in names:
        for b in names:
            if a == b:
                continue
            Xa, ya = data[a]
            Xb, yb = data[b]
            sc = StandardScaler().fit(Xa)
            clf = LogisticRegression(max_iter=1000, C=0.1).fit(sc.transform(Xa), ya)
            acc = float(clf.score(sc.transform(Xb), yb))
            chance = float(max(yb.mean(), 1 - yb.mean()))
            res["cross_%s_to_%s" % (a, b)] = {"acc": round(acc, 3), "chance": round(chance, 3)}
            print("%-14s -> %-14s accuracy %.3f   chance %.3f" % (a, b, acc, chance))

    out = Path("runs/probe-direction.json")
    out.write_text(json.dumps(res, indent=1))
    print("\nwrote %s" % out)
    print("read this as: chance ~0.5 means the features carry no left/right at all;")
    print("high within-video + low cross-video means the features encode THIS video's look.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
