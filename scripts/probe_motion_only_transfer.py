"""Is the cross-area wall caused by the model reading APPEARANCE instead of MOTION?

Measured on the same data, with the leak fixed:

  split                     V1 pixels->action   V2 pixels->displacement
  in-area (map4 -> map4)    82.7% (base 5.8%)   corr +0.989
  cross    (map4 -> map21)   3.7%              corr -0.034

Cross-area everything collapses BELOW chance, which is what confidently applying
area-specific cues somewhere they do not hold looks like.  The teacher's own
numbers transfer across areas without loss (20-22% against a 5.8% baseline), so
the failure is specifically pixels -> motion, not motion -> action.

The model is given two full frames, so it is free to solve the task by
recognising the place instead of watching what moved.  This probe removes that
option and measures what happens:

  pair     [a, b]                 the current input, appearance available
  absdiff  [|a - b|]              the motion, with the scene divided out
  sdiff    [normalised(b - a)]    the same, signed: direction survives, which is
                                  what the SIGN of a displacement needs

If the wall is appearance, absdiff/sdiff should transfer noticeably better than
pair.  If they do not, then motion alone does not carry the answer here and the
problem is upstream of the input format.

    uv run python scripts/probe_motion_only_transfer.py --train <map4...> --test <map21>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.loop.bootstrap import _prep  # noqa: E402
from ash.models.idm import IdmConfig, IdmModel  # noqa: E402
from ash.utils.device import resolve_device  # noqa: E402

SIZE = 128
BATCH = 16
LR = 3e-4
GRAD_CLIP = 1.0


def pairs_of(path: str):
    """(a, b, dpx, dpy) for one session; pairs never cross a session seam."""
    z = np.load(path, allow_pickle=True)
    frames = np.asarray(z["frames"])
    states = [json.loads(s) for s in z["states"]]
    n = len(z["acts"])
    dpx = np.zeros(n, np.float32)
    dpy = np.zeros(n, np.float32)
    for t in range(n):
        p = (states[t] or {}).get("physics") or {}
        q = (states[t + 1] or {}).get("physics") or {}
        dpx[t] = float(q.get("px") or 0) - float(p.get("px") or 0)
        dpy[t] = float(q.get("py") or 0) - float(p.get("py") or 0)
    return frames[:-1], frames[1:], dpx, dpy


def to_tensor(arr: np.ndarray, dev) -> torch.Tensor:
    """(N, H, W, C) float in [0, 1] - BHWC, because ImpalaCNN takes that layout
    and transposes internally.  Permuting to BCHW here made the trunk read the
    channel count from H: `expected input[16, 128, 3, 128] to have 3 channels`."""
    return torch.from_numpy(_prep(arr, SIZE)).to(dev).float()


def inputs_for(arm: str, a: torch.Tensor, b: torch.Tensor) -> list[torch.Tensor]:
    if arm == "pair":
        return [a, b]
    if arm == "absdiff":
        return [torch.abs(b - a)]
    if arm == "sdiff":
        # b - a lies in [-1, 1]; shift to [0, 1] so the trunk sees a normal image.
        return [(b - a + 1.0) * 0.5]
    raise ValueError(arm)


def run(dev, arm: str, tr, te, epochs: int) -> dict:
    a_tr, b_tr, dpx_tr, dpy_tr = tr
    a_te, b_te, dpx_te, dpy_te = te
    torch.manual_seed(0); np.random.seed(0)
    model = IdmModel(IdmConfig(image_size=SIZE, num_actions=20)).to(dev)
    n_in = {"pair": 2, "absdiff": 1, "sdiff": 1}[arm]
    head = nn.Linear(model.config.embed_dim * n_in, 2).to(dev)
    params = list(model.trunk.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    stacked = np.stack([dpx_tr, dpy_tr], axis=1)
    mu, sd = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6
    target = torch.from_numpy(((stacked - mu) / sd).astype(np.float32)).to(dev)

    order = np.arange(len(dpx_tr))
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        rng.shuffle(order)
        total, seen = 0.0, 0
        model.train()
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            xs = inputs_for(arm, to_tensor(a_tr[idx], dev), to_tensor(b_tr[idx], dev))
            feats = torch.cat([model.trunk(x.unsqueeze(1)).squeeze(1) for x in xs], dim=1)
            loss = nn.functional.mse_loss(head(feats), target[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            total += float(loss) * len(idx); seen += len(idx)
        print("    [%s] epoch %d train %.4f" % (arm, epoch, total / max(1, seen)), flush=True)

    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(dpx_te), 64):
            xs = inputs_for(arm, to_tensor(a_te[i : i + 64], dev), to_tensor(b_te[i : i + 64], dev))
            feats = torch.cat([model.trunk(x.unsqueeze(1)).squeeze(1) for x in xs], dim=1)
            preds.append(head(feats).cpu().numpy())
    pred = np.concatenate(preds) * sd + mu
    true = np.stack([dpx_te, dpy_te], axis=1)
    out: dict[str, float] = {}
    for j, axis in enumerate(("dpx", "dpy")):
        zero = float(((true[:, j].mean() - true[:, j]) ** 2).mean())
        mse = float(((pred[:, j] - true[:, j]) ** 2).mean())
        out[axis + "_corr"] = float(np.corrcoef(pred[:, j], true[:, j])[0, 1])
        out[axis + "_skill"] = 1.0 - mse / zero if zero > 0 else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--cap", type=int, default=6000, help="每臂训练对上限（控制耗时）")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--in-area-test", default=None, help="同区域的对照测试会话")
    args = ap.parse_args()
    dev = resolve_device("cuda")

    def load(paths: list[str], cap: int):
        cols = list(zip(*[pairs_of(p) for p in paths]))
        a = np.concatenate(cols[0]); b = np.concatenate(cols[1])
        dx = np.concatenate(cols[2]); dy = np.concatenate(cols[3])
        if cap and len(dx) > cap:                   # deterministic subset
            pick = np.random.default_rng(0).permutation(len(dx))[:cap]
            a, b, dx, dy = a[pick], b[pick], dx[pick], dy[pick]
        return a, b, dx, dy

    tr = load(args.train, args.cap)
    te = load([args.test], 0)
    print("训练 %d 对（上限 %d）| 跨区域留出 %s %d 对"
          % (len(tr[2]), args.cap, Path(args.test).name, len(te[2])))
    ref = load([args.in_area_test], 0) if args.in_area_test else None

    print("\n%6s %14s %14s %14s" % ("臂", "跨区域 dpx corr", "跨区域 dpy corr", "域内 dpx corr"))
    for arm in ("pair", "absdiff", "sdiff"):
        cross = run(dev, arm, tr, te, args.epochs)
        in_area = run(dev, arm, tr, ref, args.epochs) if ref is not None else {}
        print("%6s %14.3f %14.3f %14s"
              % (arm, cross["dpx_corr"], cross["dpy_corr"],
                 ("%.3f" % in_area["dpx_corr"]) if in_area else "-"), flush=True)
    print("\n判读：absdiff/sdiff 的跨区域 corr 明显高于 pair ⇒ 墙是外观，输入格式可以修；"
          "\n     三者都≈0 ⇒ 这份数据里运动本身不足以决定动作。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
