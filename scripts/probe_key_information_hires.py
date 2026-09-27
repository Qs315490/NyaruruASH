"""Does the corpus resolution hide the action keys?  Same CNN, two resolutions.

Every configuration tested so far read the 256x256 corpus, in which the character is about six
pixels - and the action keys (jump, attack, dash, special, ult) show up as a small, local, brief
animation on that character, while the direction keys show up as the whole background scrolling.
That asymmetry matches the results exactly: directions are readable (0.64 on a binary), action
keys never were.  So the untested question is whether the resolution is what hid them.

This trains ONE small CNN on the SAME task (multi-label key state, BCE, temporal split inside
video 1) on two inputs and changes nothing else:

    (a) the 256x256 corpus (what every earlier run used; character ~6 px)
    (b) a 512x288 corpus (2x the linear detail; character ~12 px)

If (b) lifts action keys clearly above (a), the pipeline was throwing them away and the
resolution is a real, unexplored lever.  If both land in the same place, the ~0.2 macro-F1
ceiling is not a resolution artifact and can be stated as a property of the task at 4 fps.

    uv run python scripts/probe_key_information_hires.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.utils.device import resolve_device  # noqa: E402

STEM = "BV19s4y1y7un"


def net(n_keys: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64 * 16, n_keys),
    )


def run(tag: str, frames, labels, held, dev, epochs=1, batch=32, lr=3e-4):
    """frames: mmap of (N,H,W) uint8; labels/held: (N,K) float."""
    n = min(len(frames) - 1, len(labels) - 1)
    cut = int(n * 0.7)
    model = net(labels.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    steps = max(1, cut // batch)
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for _ in range(steps):
            idx = np.sort(rng.integers(0, cut, size=batch))
            a = np.asarray(frames[idx], np.float32) / 255.0
            b = np.asarray(frames[idx + 1], np.float32) / 255.0
            d = torch.from_numpy(np.abs(a - b)).unsqueeze(1).to(dev)
            y = torch.from_numpy(labels[idx]).to(dev)
            loss = lossf(model(d), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += float(loss.detach())
        print("  %s epoch %d loss %.4f" % (tag, ep, tl / steps), flush=True)
    model.eval()
    P = []
    with torch.inference_mode():
        for i in range(cut, n, 64):
            idx = np.arange(i, min(i + 64, n))
            a = np.asarray(frames[idx], np.float32) / 255.0
            b = np.asarray(frames[idx + 1], np.float32) / 255.0
            d = torch.from_numpy(np.abs(a - b)).unsqueeze(1).to(dev)
            P.append((model(d) > 0).float().cpu().numpy())
    P = np.concatenate(P)
    Y = labels[cut:n]
    per = []
    for j in range(Y.shape[1]):
        tp = float((P[:, j] * Y[:, j]).sum()); fp = float((P[:, j] * (1 - Y[:, j])).sum())
        fn = float(((1 - P[:, j]) * Y[:, j]).sum())
        per.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    ones = Y > 0.5
    onbit = float((P[ones] == Y[ones]).mean()) if ones.any() else float("nan")
    return float(np.mean(per)), onbit, per


def main() -> int:
    dev = resolve_device("cuda")
    L = np.load("runs/keycast-labels-%s.npz" % STEM, allow_pickle=True)
    names = [str(x) for x in L["names"]]
    Y = L["held"].astype(np.float32)
    corpora = [("256x256 corpus", Path("data/corpus/%s.npy" % STEM)),
               ("512x288 hires", Path("runs/hires-%s.npy" % STEM))]
    print("%-16s %9s %9s   per-key F1" % ("input", "macroF1", "ON-bit"))
    for tag, path in corpora:
        if not path.exists():
            print("%-16s (missing %s)" % (tag, path))
            continue
        fr = np.load(path, mmap_mode="r")
        if fr.ndim == 4:                       # RGB corpus -> luminance, same detail
            fr = np.ascontiguousarray(fr.mean(axis=3).astype(np.uint8))
        f1, onbit, per = run(tag, fr, Y, None, dev)
        print("%-16s %9.3f %9.3f   %s" % (tag, f1, onbit,
              " ".join("%s=%.2f" % (k, v) for k, v in zip(names, per))))
    print("\nSame task, same net, same split - only the pixels per character differ (~6 px vs ~12 px).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
