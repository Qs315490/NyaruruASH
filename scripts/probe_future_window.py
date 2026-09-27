"""Pilot: does a FUTURE window make the ACTION keys readable?

Corrected understanding of the pipeline, which is why this is legitimate: in ASH the IDM is
trained on self-play transitions whose actions are known, and then APPLIED TO THE VIDEO CORPUS to
produce pseudo-actions (bootstrap.py: `chunks.append(idm(a, b))` over "a 13419-frame video").  Its
inference target is a RECORDING, so future frames exist and may be used.  Only the policy must stay
causal - and the policy consumes observations, not the IDM's window.

That matters because the evidence for a key press is its CONSEQUENCE: the jump arc, the attack
swing, the dash burst all unfold AFTER the press.  The corpus is 4 fps, so one pair spans 0.25 s
and cuts that consequence in half; and the earlier K sweep only ever looked BACKWARD (K difference
images of the past, peaking at K=2 and falling after).  This is the untested side.

Pilot settings, chosen to fit a 12 GB card shared with the game: +-8 frames at 30 fps (~+-0.27 s),
96x96 grayscale, small 3D CNN, one video, temporal split, one epoch.  The question is narrow:

    do the ACTION keys (jump / attack / dash / special / ult) rise from ~0?

Left/right already reach 0.45-0.47 through global background scroll, so they are not the test; if
the action keys stay at zero with a future window, the information really is not in the picture.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.utils.device import resolve_device  # noqa: E402

STEM = "BV19s4y1y7un"
W = H = 96
HALF = 8                      # +- frames at 30 fps around the tick
FPS = 30
TICK = 0.25                   # control interval, so ticks are 7.5 frames apart


def frames_30fps(video: Path) -> np.ndarray:
    out = Path("runs") / ("f30-%s.npy" % STEM)
    if out.exists():
        return np.load(out, mmap_mode="r")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", "fps=%d,scale=%d:%d" % (FPS, W, H),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    n = len(raw) // (W * H)
    a = np.frombuffer(raw, np.uint8)[:n * W * H].reshape(n, H, W)
    np.save(out, a)
    print("built %s: %d frames at %dfps" % (out, n, FPS))
    return np.load(out, mmap_mode="r")


class Net3D(nn.Module):
    def __init__(self, n_keys: int, t: int) -> None:
        super().__init__()
        self.t = t
        self.body = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=(1, 2, 2), padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=(1, 2, 2), padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=(2, 2, 2), padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(64, n_keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, T, H, W)
        x = x.unsqueeze(1)                                   # (B, 1, T, H, W)
        x = self.body(x)                                     # (B, 64, t', h', w')
        x = x.mean(dim=(2, 3, 4))
        return self.head(x)


def main() -> int:
    dev = resolve_device("cuda")
    video = Path("data/video-src/%s.mp4" % STEM)
    F = frames_30fps(video)
    L = np.load("runs/keycast-labels-%s.npz" % STEM, allow_pickle=True)
    held, names = L["held"].astype(np.float32), [str(n) for n in L["names"]]
    step = FPS * TICK                                  # 7.5 frames per tick
    n_tick = min(len(held), int(len(F) / step) - 1)
    print("30fps frames %d | ticks usable %d | window +-%d frames" % (len(F), n_tick, HALF))
    centres = (np.arange(n_tick) * step).astype(int)
    keep = (centres - HALF >= 0) & (centres + HALF < len(F))
    centres = centres[keep]
    Y = held[:n_tick][keep]
    cut = int(len(centres) * 0.7)
    print("samples %d (train %d / test %d), window length %d frames = %.2f s"
          % (len(centres), cut, len(centres) - cut, 2 * HALF + 1, (2 * HALF + 1) / FPS))

    def batch(idx):
        out = np.empty((len(idx), 2 * HALF + 1, H, W), np.float32)
        for k, c in enumerate(idx):
            out[k] = F[centres[c] - HALF:centres[c] + HALF + 1]
        return torch.from_numpy(out).div_(255.0).to(dev)

    torch.manual_seed(0)
    model = Net3D(Y.shape[1], 2 * HALF + 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    bs, steps = 16, max(1, cut // 16)
    for ep in range(1):
        model.train()
        tl = 0.0
        for i in range(steps):
            idx = np.sort(rng.integers(0, cut, size=bs))
            y = torch.from_numpy(Y[idx]).to(dev)
            loss = lossf(model(batch(idx)), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += float(loss.detach())
            if i % 100 == 0:
                print("  step %4d/%d loss %.4f" % (i, steps, tl / (i + 1)), flush=True)
        print("epoch %d train %.4f" % (ep, tl / steps), flush=True)

    model.eval()
    P = []
    with torch.inference_mode():
        for i in range(cut, len(centres), 32):
            idx = np.arange(i, min(i + 32, len(centres)))
            P.append((model(batch(idx)) > 0).float().cpu().numpy())
    P = np.concatenate(P); Yt = Y[cut:]
    per = {}
    for j, nm in enumerate(names):
        tp = float((P[:, j] * Yt[:, j]).sum()); fp = float((P[:, j] * (1 - Yt[:, j])).sum())
        fn = float(((1 - P[:, j]) * Yt[:, j]).sum())
        per[nm] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    ones = Yt > 0.5
    print("\n== FUTURE WINDOW +-%d @ %dfps (%d frames) ==" % (HALF, FPS, 2 * HALF + 1))
    print("macro-F1 %.3f  |  ON-bit %.3f" % (float(np.mean(list(per.values()))),
          float((P[ones] == Yt[ones]).mean()) if ones.any() else float("nan")))
    for nm, v in per.items():
        print("   %-8s F1 %.3f   (true positives %d)" % (nm, v, int(Yt[:, names.index(nm)].sum())))
    print("\nCompare: past-only, 4 fps -> macro-F1 0.19, action keys ~0.00, left 0.45.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
