"""Fine-tune DINOv2 ViT-S/14 as the IDM trunk - the frozen version helped, this asks how much.

The frozen run is the only lever that moved the numbers (macro-F1 0.000-0.028 -> 0.130-0.173
in two of three folds), which points at the representation.  Frozen features cannot adapt to
this domain at all, so the natural next question is whether adapting them helps - the usual
jump from frozen to fine-tuned.

Two concessions to stay affordable on a 12 GB card shared with the game:
  * 112x112 input instead of 224 - 8x8 patches instead of 16x16, roughly 4x less compute,
    and the source is pixel art upscaled to 720p, so little is lost;
  * one epoch per fold, which the frozen run showed is enough to see the direction.

Everything else matches the frozen experiment: same pairs, same [ea, eb, eb-ea] head, same
per-key macro-F1 (the metric an all-empty prediction scores 0 on).

    uv run python scripts/train_keycast_vit.py --train-stems A,B --test-stem C
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.data.video_pack import artifact  # noqa: E402

from ash.utils.device import resolve_device  # noqa: E402

_DINO_MODEL = "dinov2_vits14"
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class Idm(nn.Module):
    def __init__(self, n_keys: int, res: int = 112) -> None:
        super().__init__()
        self.res = res
        self.trunk = torch.hub.load("facebookresearch/dinov2", _DINO_MODEL, verbose=False)
        d = self.trunk.embed_dim if hasattr(self.trunk, "embed_dim") else 384
        self.head = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, 512), nn.GELU(), nn.Linear(512, n_keys),
        )
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    def embed(self, frames_u8: np.ndarray, dev) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(frames_u8)).to(dev).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.trunk(x)

    def forward(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([ea, eb, eb - ea], dim=1))


def load(stem: str):
    fr = np.load(artifact(stem, "corpus-4fps"), mmap_mode="r")
    L = np.load(artifact(stem, "labels"), allow_pickle=True)
    held, names = L["held"].astype(np.float32), [str(n) for n in L["names"]]
    return fr, held, names, min(len(fr) - 1, len(held) - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stems", required=True)
    ap.add_argument("--test-stem", required=True)
    ap.add_argument("--tag", default="vit")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--res", type=int, default=112)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="trunk lr - 2e-4 was far too high: the trunk ran away from the "
                         "pretrained weights and the model collapsed to predicting nothing")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--calib-frac", type=float, default=0.1,
                    help="tail of each TRAIN video held out to calibrate per-key thresholds")
    ap.add_argument("--pos-weight", action="store_true",
                    help="weight each key by negatives/positives (the skew fix)")
    ap.add_argument("--freeze-frac", type=float, default=0.3,
                    help="fraction of steps with the trunk frozen (head-only warmup)")
    args = ap.parse_args()

    dev = resolve_device("cuda")
    sets, keys = [], []
    for st in [x.strip() for x in args.train_stems.split(",") if x.strip()]:
        fr, held, names, n = load(st)
        sets.append((st, fr, held, names, n))
        keys += names
        print("train %-14s %d transitions" % (st, n))
    KEY_ORDER = sorted(set(keys))
    kidx = {k: i for i, k in enumerate(KEY_ORDER)}
    print("keys (%d): %s" % (len(KEY_ORDER), KEY_ORDER))

    def targets(held, names):
        y = np.zeros((len(held), len(KEY_ORDER)), np.float32)
        for j, nm in enumerate(names):
            if nm in kidx:
                y[:, kidx[nm]] = held[:, j]
        return y

    # Split each training video's transitions: the head of it trains, the tail calibrates
    # the per-key thresholds.  The TEST video is never touched for this.
    train, calib = [], []
    for _st, fr, h, nm, n in sets:
        y = targets(h, nm)
        cut = max(1, int(n * (1.0 - args.calib_frac)))
        train.append((fr, y, cut))
        calib.append((fr, y, cut, n))
    tot = sum(t[2] for t in train)
    weights = np.array([t[2] / tot for t in train])
    t_fr, t_held, t_names, t_n = load(args.test_stem)
    tY = targets(t_held, t_names)
    print("test  %-14s %d transitions" % (args.test_stem, t_n))

    torch.manual_seed(0)
    model = Idm(len(KEY_ORDER), args.res).to(dev)
    head = [p for n, p in model.named_parameters() if n.startswith("head")]
    trunk = [p for n, p in model.named_parameters() if not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": trunk, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=1e-4)
    if args.pos_weight:
        pos = np.zeros(len(KEY_ORDER), np.float64)
        seen = 0
        for _fr, y, cut in train:
            pos += y[:cut].sum(axis=0); seen += cut
        pw = np.clip((seen - pos) / np.maximum(pos, 1.0), 1.0, 20.0)
        print("pos_weight: %s" % dict(zip(KEY_ORDER, pw.round(2))))
        lossf = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pw, dtype=torch.float32, device=dev))
    else:
        lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    total_steps = max(1, args.epochs * max(1, tot // args.batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    # Freeze the trunk for the first stretch so the head can settle before the pretrained
    # weights are allowed to move at all.
    for prm in trunk:
        prm.requires_grad = False
    thaw_at = int(total_steps * args.freeze_frac)
    for ep in range(args.epochs):
        model.train()
        steps = max(1, tot // args.batch)
        tl = 0.0
        for i in range(steps):
            vi = int(rng.choice(len(train), p=weights))
            fr, y, n = train[vi]
            idx = np.sort(rng.integers(0, n, size=args.batch))
            ea = model.embed(fr[idx], dev)
            eb = model.embed(fr[idx + 1], dev)
            loss = lossf(model(ea, eb), torch.from_numpy(y[idx]).to(dev))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tl += float(loss.detach())
            gstep = ep * steps + i
            if gstep == thaw_at:
                for prm in trunk:
                    prm.requires_grad = True
                print("  trunk unfrozen at step %d (lr %.2e)" % (gstep, args.lr), flush=True)
            if i % 100 == 0:
                print("  epoch %d step %4d/%d loss %.4f trunk_frozen=%s"
                      % (ep, i, steps, tl / (i + 1), not trunk[0].requires_grad), flush=True)
        print("epoch %d train %.4f" % (ep, tl / steps), flush=True)

    model.eval()

    def logits_of(fr, start, stop, step=128):
        out = []
        with torch.inference_mode():
            for i in range(start, stop, step):
                idx = np.arange(i, min(i + step, stop))
                out.append(model(model.embed(fr[idx], dev), model.embed(fr[idx + 1], dev)))
        return torch.cat(out) if out else torch.zeros(0, len(KEY_ORDER), device=dev)

    # Per-key thresholds, chosen on the training videos' tails by F1.  The baseline 0 is
    # what a balanced loss would give; the skew fix is letting each key pick its own.
    THR = np.zeros(len(KEY_ORDER), np.float32)
    grid = np.arange(-4.0, 4.01, 0.25, dtype=np.float32)
    C = torch.cat([logits_of(fr, cut, n) for fr, y, cut, n in calib]).cpu().numpy()
    CY = np.concatenate([y[cut:n] for _fr, y, cut, n in calib]).astype(bool)
    for j in range(len(KEY_ORDER)):
        best, bt = -1.0, 0.0
        for t in grid:
            pr = C[:, j] > t
            tp = float((pr & CY[:, j]).sum()); fp = float((pr & ~CY[:, j]).sum())
            fn = float((~pr & CY[:, j]).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best:
                best, bt = f1, float(t)
        THR[j] = bt
    print("calibrated thresholds: %s"
          % dict(zip(KEY_ORDER, THR.round(2))), flush=True)

    T = logits_of(t_fr, 0, t_n).cpu().numpy()
    P = T > THR[None, :]
    Y = tY[:t_n]
    per = {}
    for k, j in kidx.items():
        tp = float((P[:, j] * Y[:, j]).sum()); fp = float((P[:, j] * (1 - Y[:, j])).sum())
        fn = float(((1 - P[:, j]) * Y[:, j]).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[k] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
                  "true_positives": int(Y[:, j].sum())}
    ones = Y > 0.5
    macro = float(np.mean([v["f1"] for v in per.values()]))
    res = {"backbone": "dinov2_vits14 fine-tuned", "res": args.res, "epochs": args.epochs,
           "trunk_lr": args.lr, "freeze_frac": args.freeze_frac,
           "pos_weight": bool(args.pos_weight), "calib_frac": args.calib_frac,
           "thresholds": [round(float(x), 2) for x in THR],
           "train": [s[0] for s in sets], "test": args.test_stem,
           "macro_f1": round(macro, 3), "bit_accuracy": round(float((P == Y).mean()), 4),
           "accuracy_on_active_bits": round(float((P[ones] == Y[ones]).mean()), 4) if ones.any() else None,
           "exact_set_match": round(float((P == Y).all(axis=1).mean()), 4),
           "pred_positive_rate": round(float(P.mean()), 4),
           "true_positive_rate": round(float(Y.mean()), 4), "per_key": per}
    print("\n== FINE-TUNED ViT-S @%d: train %s -> test %s =="
          % (args.res, [s[0] for s in sets], args.test_stem))
    print("macro-F1 %.3f  (all-empty scores 0.000)" % macro)
    print("per-bit accuracy %.1f%%  |  on bits ON in the truth: %s"
          % (100 * res["bit_accuracy"],
             "%.1f%%" % (100 * res["accuracy_on_active_bits"])
             if res["accuracy_on_active_bits"] is not None else "n/a"))
    print("exact key-set match %.1f%%  |  predicted positive rate %.1f%% vs true %.1f%%"
          % (100 * res["exact_set_match"], 100 * res["pred_positive_rate"],
             100 * res["true_positive_rate"]))
    for k, v in per.items():
        print("   %-8s P %.3f  R %.3f  F1 %.3f  (true positives %d)"
              % (k, v["precision"], v["recall"], v["f1"], v["true_positives"]))
    out = Path("runs") / ("idm-vit-%s-%s.json" % (args.test_stem, args.tag))
    out.write_text(json.dumps(res, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
