"""Synthetic domains: does the cross-video gap shrink when the source videos span many domains?

The three labelled videos differ in resolution, encoder, overlay layout and game-window position.
Only three of them exist, so the "collect more videos" lever is closed.  But the differences are
exactly the kind that can be SIMULATED, which turns three videos into many domains:

    blur (0.8 / 1.6 sigma)      a softer, lower-detail capture
    jpeg 30 / 60                encoder artifacts
    zoom 90% / 80%              different framing and game-window scale
    bright 0.8 / 1.25           exposure and post-processing
    down 48 then up to 96       a genuinely lower-resolution source

This differs from the earlier augmentation attempt in kind, not in degree: that applied a mild
per-sample jitter, whereas this manufactures whole video-level domains.  If cross-video failure is
"too few domains", training across these should close part of the gap; if it is something else,
this moves nothing.

Features are cached per (video, domain) because materialising 27 stores at 30 fps would need 25 GB.

    uv run python scripts/build_synthetic_domains.py --stage cache
    uv run python scripts/build_synthetic_domains.py --stage train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.data.video_pack import artifact  # noqa: E402
from ash.utils.device import resolve_device  # noqa: E402

STEMS = ["BV19s4y1y7un", "BV13HnzzPEEN", "BV1hc411M7GW"]
ACTIONS = ["jump", "attack", "dash", "special", "ult"]
STEP, HALF, R = 7.5, 8, 16                 # 30 fps, +-8 future frames, 16x16 grid
CAP = 4000
S = 96 // R


def domains():
    """name -> callable(frame uint8 96x96) -> frame uint8 96x96"""
    def blur(sig):
        return lambda a: cv2.GaussianBlur(a, (0, 0), sig)

    def jpeg(q):
        def f(a):
            ok, buf = cv2.imencode(".jpg", a, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else a
        return f

    def zoom(f_):
        def f(a):
            s = int(round(96 * f_))
            o = (96 - s) // 2
            return cv2.resize(a[o:o + s, o:o + s], (96, 96), interpolation=cv2.INTER_AREA)
        return f

    def gain(g):
        return lambda a: np.clip(a.astype(np.float32) * g, 0, 255).astype(np.uint8)

    def down(scale):
        def f(a):
            return cv2.resize(cv2.resize(a, (scale, scale), interpolation=cv2.INTER_AREA),
                              (96, 96), interpolation=cv2.INTER_LINEAR)
        return f

    return {
        "id": lambda a: a,
        "blur08": blur(0.8), "blur16": blur(1.6),
        "jpeg60": jpeg(60), "jpeg30": jpeg(30),
        "zoom90": zoom(0.90), "zoom80": zoom(0.80),
        "bright08": gain(0.8), "bright125": gain(1.25),
        "down48": down(48),
    }


def union_keys() -> list:
    """All key names across the videos, in one order.

    Videos carry different key sets (one has 7, the others 9), so a per-video label matrix cannot be
    concatenated with another's - the columns mean different keys.  Everything below uses this order
    and zero-fills the keys a video does not have.
    """
    ks = set()
    for st in STEMS:
        p = artifact(st, "labels")
        ks |= {str(n) for n in np.load(p, allow_pickle=True)["names"]}
    return sorted(ks)


def wide_y(stem: str, keys: list) -> np.ndarray:
    L = np.load(artifact(stem, "labels"), allow_pickle=True)
    held, names = L["held"].astype(np.float32), [str(n) for n in L["names"]]
    Y = np.zeros((len(held), len(keys)), np.float32)
    for j, nm in enumerate(names):
        if nm in keys:
            Y[:, keys.index(nm)] = held[:, j]
    return Y


def tick_filter(stem: str) -> np.ndarray:
    """Row indices of `wide_y` that correspond to the cached features (same CAP/keep rule)."""
    F = np.load(artifact(stem, "frames-30fps-gamearea"), mmap_mode="r")
    held = np.load(artifact(stem, "labels"), allow_pickle=True)["held"]
    n_tick = min(len(held), int(len(F) / STEP) - 1)
    c = (np.arange(n_tick) * STEP).astype(int)
    keep = (c - HALF - 8 >= 0) & (c + HALF + 8 < len(F))
    idx = np.where(keep)[0]
    if len(idx) > CAP:
        idx = idx[np.linspace(0, len(idx) - 1, CAP).astype(int)]
    return idx


def cache_path(stem: str, dom: str) -> Path:
    return Path("runs") / ("dom-%s-%s.npy" % (stem, dom))


def build_cache(stem: str) -> None:
    src = artifact(stem, "frames-30fps-gamearea")
    labels = artifact(stem, "labels")
    if src is None or labels is None:
        print("  %s: missing frames/labels" % stem)
        return
    F = np.load(src, mmap_mode="r")
    L = np.load(labels, allow_pickle=True)
    held = L["held"].astype(np.float32)
    names = [str(n) for n in L["names"]]
    n_tick = min(len(held), int(len(F) / STEP) - 1)
    c = (np.arange(n_tick) * STEP).astype(int)
    keep = (c - HALF - 8 >= 0) & (c + HALF + 8 < len(F))
    c, held = c[keep], held[:n_tick][keep]
    if len(c) > CAP:
        sel = np.linspace(0, len(c) - 1, CAP).astype(int)
        c, held = c[sel], held[sel]
    Y = np.zeros((len(held), len(names)), np.float32)
    for j in range(len(names)):
        Y[:, j] = held[:, j]
    D = domains()
    for dom, fn in D.items():
        out = cache_path(stem, dom)
        if out.exists():
            continue
        X = np.empty((len(c), 9 * R * R), np.float32)
        for i in range(0, len(c), 32):
            cc = c[i:i + 32]
            cols = []
            for o in range(HALF + 1):
                blk = np.asarray(F[cc + o])                    # (m, 96, 96)
                tr = np.stack([fn(blk[k]) for k in range(len(cc))]).astype(np.float32)
                cols.append(tr.reshape(len(cc), R, S, R, S).mean(axis=(2, 4)).reshape(len(cc), -1))
            X[i:i + len(cc)] = np.concatenate(cols, axis=1)
        np.save(out, X)
        np.save(Path("runs") / ("dom-%s-Y.npy" % stem), Y)
        print("  %-14s %-9s %s" % (stem, dom, X.shape), flush=True)


class MLP(nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, n_out))

    def forward(self, x):
        return self.net(x)


def fit_eval(Xtr, Ytr, Xte, Yte, dev, epochs=12, bs=256, lr=1e-3):
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
    return float(np.mean(f1s)) if f1s else 0.0


def train_stage() -> int:
    dev = resolve_device("cuda")
    D = list(domains())
    keys = union_keys()
    Ys = {s: wide_y(s, keys)[tick_filter(s)] for s in STEMS}
    print("keys (%d): %s" % (len(keys), keys))
    print("%-34s %9s   %s" % ("train -> test", "macroF1", "condition"))
    for test in STEMS:
        tr = [s for s in STEMS if s != test]
        for cond, doms in (("originals only", ["id"]), ("originals + 9 domains", D)):
            Xtr = np.concatenate([np.load(cache_path(s, d)) for s in tr for d in doms])
            Ytr = np.concatenate([Ys[s] for s in tr for _ in doms])
            Xte = np.load(cache_path(test, "id"))
            f1 = fit_eval(Xtr, Ytr, Xte, Ys[test], dev)
            print("%-34s %9.3f   %s"
                  % ("%s -> %s" % ("+".join(x[:8] for x in tr), test[:8]), f1, cond), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="cache", choices=["cache", "train"])
    a = ap.parse_args()
    if a.stage == "cache":
        for s in STEMS:
            build_cache(s)
        print("caches done")
    else:
        return train_stage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
