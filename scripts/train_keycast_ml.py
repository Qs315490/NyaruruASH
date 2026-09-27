"""IDM with a per-key multi-label objective instead of 27-way classification.

Why: every cross-video fold so far collapsed to "predict the majority class" - the 27-way
softmax over labels whose noop share is 30-77% makes that the cheapest way down, and the
predictions (92-99% noop) say exactly that.  A per-key sigmoid cannot win by ignoring the
rare keys: each key is scored on its own, so an always-empty prediction scores F1 = 0 for
every key instead of a respectable accuracy.

This is the same data, splits and capacity as the classification control - only the output
head and the loss change.  Reported per key: precision / recall / F1, plus bit accuracy and
"exact key set" match.  Macro-F1 over keys is the headline number, because it is the one an
all-empty prediction cannot fake.

    uv run python scripts/train_keycast_ml.py --train-stems A,B --test-stem C
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

from ash.models.idm import IdmConfig, IdmModel  # noqa: E402
from ash.utils.device import resolve_device  # noqa: E402


def load(stem: str):
    """(frames mmap, held bits, key names) for one video, aligned by construction."""
    fr = np.load(Path("data/corpus/%s.npy" % stem), mmap_mode="r")
    L = np.load(Path("runs/keycast-labels-%s.npz" % stem), allow_pickle=True)
    held = L["held"].astype(np.float32)
    names = [str(n) for n in L["names"]]
    n = min(len(fr) - 1, len(held) - 1)
    return fr, held, names, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stems", required=True)
    ap.add_argument("--test-stem", required=True)
    ap.add_argument("--tag", default="ml")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--diff-input", action="store_true",
                    help="feed |a-b| into the trunk AS AN IMAGE, instead of the pooled "
                         "[ea, eb, eb-ea] the other runs used.  AGENTS.md §5 measured 49.0%% "
                         "for the pooled subtraction and 85.7%% for the difference image; the "
                         "linear probe here showed the pooled feature difference has no "
                         "left/right information at all while the pixel difference does.")
    args = ap.parse_args()

    train_stems = [x.strip() for x in args.train_stems.split(",") if x.strip()]
    sets = []
    for st in train_stems:
        fr, held, names, n = load(st)
        sets.append((st, fr, held, names, n))
        print("train %-14s %d transitions, %d keys" % (st, n, held.shape[1]))
    KEY_ORDER = sorted({k for s in sets for k in s[3]})
    print("keys (%d): %s" % (len(KEY_ORDER), KEY_ORDER))
    kidx = {k: i for i, k in enumerate(KEY_ORDER)}

    def targets(held, names):
        """Wide multi-hot targets: a key absent from this video is always 0."""
        y = np.zeros((held.shape[0], len(KEY_ORDER)), np.float32)
        for j, nm in enumerate(names):
            if nm in kidx:
                y[:, kidx[nm]] = held[:, j]
        return y

    train = [(fr, targets(held, names), n) for _st, fr, held, names, n in sets]
    tot = sum(t[2] for t in train)
    weights = np.array([t[2] / tot for t in train])
    t_fr, t_held, t_names, t_n = load(args.test_stem)
    t_y = targets(t_held, t_names)
    print("test  %-14s %d transitions" % (args.test_stem, t_n))

    dev = resolve_device("cuda")

    def pair(src, idx):
        a = torch.from_numpy(np.asarray(src[idx], np.float32) / 255.0).to(dev).permute(0, 3, 1, 2)
        b = torch.from_numpy(np.asarray(src[idx + 1], np.float32) / 255.0).to(dev).permute(0, 3, 1, 2)
        if args.size != a.shape[-1]:
            a = nn.functional.interpolate(a, size=(args.size, args.size), mode="bilinear",
                                          align_corners=False)
            b = nn.functional.interpolate(b, size=(args.size, args.size), mode="bilinear",
                                          align_corners=False)
        if args.diff_input:
            # The difference image is what the trunk sees; the second slot is a constant so
            # the model's own [ea, eb, eb-ea] head still works unchanged.  Every trainable
            # path that matters then reads the motion rather than each frame's appearance.
            d = (a - b).abs()
            return d.permute(0, 2, 3, 1), torch.zeros_like(d).permute(0, 2, 3, 1)
        return a.permute(0, 2, 3, 1), b.permute(0, 2, 3, 1)      # BHWC

    torch.manual_seed(0)
    model = IdmModel(IdmConfig(image_size=args.size, num_actions=len(KEY_ORDER))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        model.train()
        steps = max(1, tot // args.batch)
        tot_loss = 0.0
        for _ in range(steps):
            vi = int(rng.choice(len(train), p=weights))
            fr, y, n = train[vi]
            idx = np.sort(rng.integers(0, n, size=args.batch))
            a, b = pair(fr, idx)
            logits = model(a, b)
            loss = lossf(logits, torch.from_numpy(y[idx]).to(dev))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += float(loss.detach())
        print("epoch %d train %.4f" % (ep, tot_loss / steps), flush=True)

    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, t_n, 128):
            idx = np.arange(i, min(i + 128, t_n))
            a, b = pair(t_fr, idx)
            preds.append((model(a, b) > 0).float().cpu().numpy())
    P = np.concatenate(preds)
    Y = t_y[:t_n]
    per = {}
    for k, j in kidx.items():
        tp = float((P[:, j] * Y[:, j]).sum())
        fp = float((P[:, j] * (1 - Y[:, j])).sum())
        fn = float(((1 - P[:, j]) * Y[:, j]).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[k] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
                  "true_positives": int(Y[:, j].sum())}
    macro_f1 = float(np.mean([v["f1"] for v in per.values()]))
    ones = Y > 0.5
    res = {
        "diff_input": bool(args.diff_input),
        "train": train_stems, "test": args.test_stem, "keys": KEY_ORDER,
        "macro_f1": round(macro_f1, 3),
        "bit_accuracy": round(float((P == Y).mean()), 4),
        "accuracy_on_active_bits": round(float((P[ones] == Y[ones]).mean()), 4) if ones.any() else None,
        "exact_set_match": round(float((P == Y).all(axis=1).mean()), 4),
        "pred_positive_rate": round(float(P.mean()), 4),
        "true_positive_rate": round(float(Y.mean()), 4),
        "per_key": per,
    }
    print("\n== MULTI-LABEL%s: train %s -> test %s =="
          % (" |a-b| INPUT" if args.diff_input else "", train_stems, args.test_stem))
    print("macro-F1 %.3f  (an all-empty prediction scores 0.000)" % macro_f1)
    print("per-bit accuracy %.1f%%  |  on bits that are ON in the truth: %s"
          % (100 * res["bit_accuracy"],
             "%.1f%%" % (100 * res["accuracy_on_active_bits"])
             if res["accuracy_on_active_bits"] is not None else "n/a"))
    print("exact key-set match %.1f%%  |  predicted positive rate %.1f%% vs true %.1f%%"
          % (100 * res["exact_set_match"], 100 * res["pred_positive_rate"],
             100 * res["true_positive_rate"]))
    for k, v in per.items():
        print("   %-8s P %.3f  R %.3f  F1 %.3f   (true positives %d)"
              % (k, v["precision"], v["recall"], v["f1"], v["true_positives"]))
    out = Path("runs") / ("idm-multilabel-%s-%s.json" % (args.test_stem, args.tag))
    out.write_text(json.dumps(res, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
