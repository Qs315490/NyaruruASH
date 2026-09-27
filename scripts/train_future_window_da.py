"""Explicit domain adaptation: CORAL, MMD, and adversarial (DANN).

Three interventions have failed to move cross-video transfer (multi-video training, augmentation,
pipeline normalisation), while within a video the same features reach 0.59-0.73 macro-F1.  That is
the signature of a domain gap, so this applies the standard unsupervised-domain-adaptation toolbox:
the held-out video's features are used WITHOUT ITS LABELS to align the representations, and its
labels are used only for the final score.

    CORAL   penalise the difference in second-order statistics (batch covariance) between domains
    MMD     penalise the maximum mean discrepancy between the feature distributions (RBF kernel)
    DANN    a domain classifier behind a gradient-reversal layer, so the trunk is pushed to make
            the domains indistinguishable

Same folds, same features, same head, same metric as the plain baseline (0.015 / 0.118 / 0.004), so
any movement is readable.

    uv run python scripts/train_future_window_da.py --method coral
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Net(nn.Module):
    def __init__(self, d_in: int, n_out: int, n_dom: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU())
        self.head = nn.Linear(256, n_out)
        self.dom = nn.Linear(256, n_dom)

    def forward(self, x):
        h = self.trunk(x)
        return h, self.head(h), self.dom(h)


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def coral(hs, ht):
    d = hs.shape[1]
    cs = torch.cov(hs.T) + torch.eye(d, device=hs.device) * 1e-4
    ct = torch.cov(ht.T) + torch.eye(d, device=hs.device) * 1e-4
    return ((cs - ct) ** 2).sum() / (4 * d * d)


def mmd(hs, ht, sigma=1.0):
    def k(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return torch.exp(-d2 / (2 * sigma * sigma))
    return k(hs, hs).mean() + k(ht, ht).mean() - 2 * k(hs, ht).mean()


def load(stem, keys):
    F = np.load(_path(stem, "frames-30fps"), mmap_mode="r")
    L = np.load(_path(stem, "labels",), allow_pickle=True)
    held, names = L["held"].astype(np.float32), [str(n) for n in L["names"]]
    n_tick = min(len(held), int(len(F) / STEP) - 1)
    c = (np.arange(n_tick) * STEP).astype(int)
    k = (c - HALF - 6 >= 0) & (c + HALF + 6 < len(F))
    c, held = c[k], held[:n_tick][k]
    X = np.empty((len(c), 9 * R * R), np.float32)
    for i in range(0, len(c), 48):
        cc = c[i:i + 48]
        X[i:i + len(cc)] = np.concatenate([
            np.asarray(F[cc + o], np.float32).reshape(len(cc), R, S, R, S)
              .mean(axis=(2, 4)).reshape(len(cc), -1) for o in range(HALF + 1)], axis=1)
    Y = np.zeros((len(held), len(keys)), np.float32)
    for j, nm in enumerate(names):
        if nm in keys:
            Y[:, keys.index(nm)] = held[:, j]
    return X, Y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="coral", choices=["none", "coral", "mmd", "dann"])
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    dev = resolve_device("cuda")
    keys = sorted({str(n) for s in STEMS for n in np.load(
        _path(s, "labels"), allow_pickle=True)["names"].tolist()})
    data = {s: load(s, keys) for s in STEMS}
    print("method: %s (weight %.2f)" % (args.method, args.weight))
    print("%-30s %9s %11s" % ("train -> test", "macroF1", "action keys"))
    for test in STEMS:
        tr = [s for s in STEMS if s != test]
        Xtr = np.concatenate([data[s][0] for s in tr]).astype(np.float32)
        Ytr = np.concatenate([data[s][1] for s in tr])
        Xte, Yte = data[test][0].astype(np.float32), data[test][1]
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        xt = torch.from_numpy(Xtr).to(dev); yt = torch.from_numpy(Ytr).to(dev)
        xv = torch.from_numpy(Xte).to(dev)                      # target features, NO labels
        m = Net(Xtr.shape[1], Ytr.shape[1], 2).to(dev)   # 2 domain labels: source/target
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(0)
        for ep in range(args.epochs):
            m.train()
            for _ in range(max(1, len(xt) // 256)):
                si = rng.integers(0, len(xt), 256)
                ti = rng.integers(0, len(xv), 256)
                hs, logits, dlogits = m(xt[si])
                ht, _, _ = m(xv[ti])
                loss = bce(logits, yt[si])
                if args.method == "coral":
                    loss = loss + args.weight * coral(hs, ht)
                elif args.method == "mmd":
                    loss = loss + args.weight * mmd(hs, ht)
                if args.method == "dann":
                    # Domain classifier on source-vs-target features, behind a gradient
                    # reversal so the trunk is pushed to make the two indistinguishable.
                    lam = args.weight * (ep / max(1, args.epochs - 1))
                    xc = torch.cat([xt[si], xv[ti]], 0)
                    dom_y = torch.cat([torch.zeros(len(si)), torch.ones(len(ti))]).long().to(dev)
                    dom_logits = m.dom(GradRev.apply(m.trunk(xc), lam))
                    loss = bce(logits, yt[si]) + F.cross_entropy(dom_logits, dom_y)
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.inference_mode():
            P = (m(xv)[1] > 0).float().cpu().numpy()
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
        print("%-30s %9.3f %11.3f"
              % ("%s -> %s" % ("+".join(x[:8] for x in tr), test[:8]),
                 float(np.mean(f1s)), float(np.mean(act)) if act else float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
