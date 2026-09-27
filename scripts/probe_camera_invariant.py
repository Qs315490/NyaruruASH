"""Is there a camera-INVARIANT quantity the pixels can predict across areas?

The wall measured so far: within one area the pixels predict the player's
displacement almost perfectly (corr 0.97-0.99), and across areas the correlation
is zero or negative - for the action target, for the displacement target, and
for motion-only inputs alike.  What each of those shares is that the TARGET is in
WORLD coordinates, and the amount of screen movement that corresponds to a metre
of world movement depends on whether the camera follows the player in that area.

So ask for the quantity that the camera cannot contaminate:

    world    diff -> (dpx, dpy)                     world displacement, as before
    screen   diff -> (dscreenX, dscreenY)           movement in SCREEN space

If `screen` transfers while `world` does not, the representation is fine and the
formulation was wrong: world displacement is then screen displacement plus camera
motion, and camera motion is a property of the video that can be estimated from
it.  If `screen` fails too, the wall is deeper than the coordinate frame.

    uv run python scripts/probe_camera_invariant.py --train <map4...> --test <map21> \
        --in-area-test <map4 session>
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
EPOCHS = 6
GRAD_CLIP = 1.0


def pairs_of(path: str):
    """(diff_inputs, world target, screen target) for one session.

    Pairs are built inside the session, so no transition spans a seam.  The
    screen target comes from `player.screenX/screenY`, which is the position
    signal this game actually updates (its tile coordinates and `_realX/_realY`
    are frozen - reading those has produced three wrong conclusions here).
    """
    z = np.load(path, allow_pickle=True)
    frames = np.asarray(z["frames"])
    states = [json.loads(s) for s in z["states"]]
    n = len(z["acts"])
    world = np.zeros((n, 2), np.float32)
    screen = np.zeros((n, 2), np.float32)
    for t in range(n):
        p = (states[t] or {}).get("physics") or {}
        q = (states[t + 1] or {}).get("physics") or {}
        pl = (states[t] or {}).get("player") or {}
        ql = (states[t + 1] or {}).get("player") or {}
        world[t] = (float(q.get("px") or 0) - float(p.get("px") or 0),
                    float(q.get("py") or 0) - float(p.get("py") or 0))
        screen[t] = (float(ql.get("screenX") or 0) - float(pl.get("screenX") or 0),
                     float(ql.get("screenY") or 0) - float(pl.get("screenY") or 0))
    return frames[:-1], frames[1:], world, screen


def to_tensor(arr, dev):
    return torch.from_numpy(_prep(arr, SIZE)).to(dev).float()   # BHWC for ImpalaCNN


def run(dev, arm: str, tr, te, epochs: int) -> dict:
    a_tr, b_tr, w_tr, s_tr = tr
    a_te, b_te, w_te, s_te = te
    y_tr = w_tr if arm == "world" else s_tr
    y_te = w_te if arm == "world" else s_te
    torch.manual_seed(0); np.random.seed(0)
    model = IdmModel(IdmConfig(image_size=SIZE, num_actions=20)).to(dev)
    head = nn.Linear(model.config.embed_dim, 2).to(dev)
    params = list(model.trunk.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    mu, sd = y_tr.mean(axis=0), y_tr.std(axis=0) + 1e-6
    target = torch.from_numpy(((y_tr - mu) / sd).astype(np.float32)).to(dev)

    order = np.arange(len(y_tr))
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        rng.shuffle(order)
        total, seen = 0.0, 0
        model.train()
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            x = torch.abs(to_tensor(b_tr[idx], dev) - to_tensor(a_tr[idx], dev))
            feats = model.trunk(x.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(head(feats), target[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            total += float(loss) * len(idx); seen += len(idx)
        print("    [%s] epoch %d train %.4f" % (arm, epoch, total / max(1, seen)), flush=True)

    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(y_te), 64):
            x = torch.abs(to_tensor(b_te[i : i + 64], dev) - to_tensor(a_te[i : i + 64], dev))
            preds.append(head(model.trunk(x.unsqueeze(1)).squeeze(1)).cpu().numpy())
    pred = np.concatenate(preds) * sd + mu
    out = {}
    for j, axis in enumerate(("x", "y")):
        zero = float(((y_te[:, j].mean() - y_te[:, j]) ** 2).mean())
        mse = float(((pred[:, j] - y_te[:, j]) ** 2).mean())
        out[axis + "_corr"] = float(np.corrcoef(pred[:, j], y_te[:, j])[0, 1])
        out[axis + "_skill"] = 1.0 - mse / zero if zero > 0 else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--in-area-test", default=None)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    dev = resolve_device("cuda")

    def load(paths, cap):
        cols = list(zip(*[pairs_of(p) for p in paths]))
        a = np.concatenate(cols[0]); b = np.concatenate(cols[1])
        w = np.concatenate(cols[2]); sc = np.concatenate(cols[3])
        if cap and len(w) > cap:
            pick = np.random.default_rng(0).permutation(len(w))[:cap]
            a, b, w, sc = a[pick], b[pick], w[pick], sc[pick]
        return a, b, w, sc

    tr = load(args.train, args.cap)
    te = load([args.test], 0)
    ref = load([args.in_area_test], 0) if args.in_area_test else None
    print("训练 %d 对 | 跨区域留出 %s %d 对" % (len(tr[2]), Path(args.test).name, len(te[2])))
    print("留出集目标尺度：world std %s | screen std %s"
          % (np.round(te[2].std(axis=0), 2), np.round(te[3].std(axis=0), 2)))
    print("\n%8s %16s %16s %16s" % ("目标", "跨区域 corr_x", "跨区域 corr_y", "域内 corr_x"))
    for arm in ("world", "screen"):
        cross = run(dev, arm, tr, te, args.epochs)
        ina = run(dev, arm, tr, ref, args.epochs) if ref is not None else {}
        print("%8s %16.3f %16.3f %16s"
              % (arm, cross["x_corr"], cross["y_corr"],
                 ("%.3f" % ina["x_corr"]) if ina else "-"), flush=True)
    print("\n判读（对照两臂的跨区域 corr）："
          "\n   world 明显更高 ⇒ 世界位移是更可迁移的框架（物理量），覆盖到位时它跨区域成立；"
          "\n   screen 明显更高 ⇒ 屏幕位移（相机不变）更可迁移，世界位移再由相机运动补回；"
          "\n   两者都≈0 ⇒ 这一组训练区域不足以支撑迁移，先加覆盖再谈配方。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
