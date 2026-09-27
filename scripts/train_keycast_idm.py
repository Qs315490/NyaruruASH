"""Train the IDM on the speedrun videos' REAL labels, honestly split.

Every IDM result in docs/status.md was measured against PSEUDO-labels: the IDM's own
predictions on the corpus, which is why "50% noop" was unfalsifiable.  The keycast
extraction changed that - the video carries its own action labels, aligned frame for
frame with the corpus - so this is the first run where the supervision is ground
truth rather than a guess.

The split is temporal (train on the first part, test on the last), because a random
split leaks near-duplicate neighbours across the boundary and that is what produced
the retracted 91.6%.

    uv run python scripts/train_keycast_idm.py --stem BV19s4y1y7un --epochs 6
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

from ash.actions.space import DEFAULT_NUM_ACTIONS  # noqa: E402
from ash.models.idm import IdmConfig, IdmModel  # noqa: E402
from ash.utils.device import resolve_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="BV19s4y1y7un")
    ap.add_argument("--train-stems", default=None,
                    help="comma-separated videos to train on (overrides --stem)")
    ap.add_argument("--test-stem", default=None,
                    help="train on --stem and evaluate on this OTHER video (cross-video)")
    ap.add_argument("--frames", default=None, help="corpus npy (default data/corpus/<stem>.npy)")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-frac", type=float, default=0.7, help="first fraction, temporal")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--augment", action="store_true",
                    help="colour jitter + paired random crop/scale (train only)")
    args = ap.parse_args()

    frames_path = Path(args.frames or ("data/corpus/%s.npy" % args.stem))
    labels_path = Path(args.labels or ("runs/keycast-labels-%s.npz" % args.stem))
    frames = np.load(frames_path, mmap_mode="r")
    labels = np.load(labels_path, allow_pickle=True)
    masks = labels["masks"].astype(np.int64)
    n = min(len(frames) - 1, len(masks) - 1)
    print("train: frames %d | labels %d | using %d transitions"
          % (len(frames), len(masks), n))

    # "cross" means an explicit test video was named - NOT "test differs from train".
    # Asking for the same name is legitimate (it keeps the output filename meaningful) and
    # treating that as a temporal split left t_frames undefined at evaluation time.
    cross = bool(args.test_stem) or bool(args.train_stems)
    t_stem = args.test_stem or args.stem
    print("cross=%s train_stems=%s test_stem=%s" % (cross, args.train_stems, args.test_stem))
    if cross:
        # Cross-video: the test set is a DIFFERENT runner's capture, so this measures
        # whether the labels describe the game rather than the video they came from.
        # Training uses the whole train video (no split) since the separation is the point.
        t_frames = np.load(Path("data/corpus/%s.npy" % args.test_stem), mmap_mode="r")
        t_labels = np.load(Path("runs/keycast-labels-%s.npz" % args.test_stem),
                           allow_pickle=True)
        t_masks = t_labels["masks"].astype(np.int64)
        tn = min(len(t_frames) - 1, len(t_masks) - 1)
        print("test : %s frames %d | labels %d | using %d transitions"
              % (t_stem, len(t_frames), len(t_masks), tn))
        tr = np.arange(0, n)
        te = np.arange(0, tn)
    else:
        cut = int(n * args.train_frac)
        tr = np.arange(0, cut)
        te = np.arange(cut, n)
    # Multi-video training: each batch is drawn from ONE video, chosen in proportion to
    # its length.  That keeps the per-video indexing vectorised and gives every video its
    # share of the gradient without materialising a merged copy of the corpora (which
    # would be 3 GB per video on a disk that is already tight).
    multi = None
    if args.train_stems:
        multi = []
        for st in [x.strip() for x in args.train_stems.split(",") if x.strip()]:
            fr = np.load(Path("data/corpus/%s.npy" % st), mmap_mode="r")
            mk = np.load(Path("runs/keycast-labels-%s.npz" % st),
                         allow_pickle=True)["masks"].astype(np.int64)
            n_i = min(len(fr) - 1, len(mk) - 1)     # NOT `nn`: that shadows torch.nn
            multi.append((st, fr, mk, n_i))
            print("train %-14s %d transitions" % (st, n_i))
        tot = sum(m[3] for m in multi)
        weights = np.array([m[3] / tot for m in multi])
        print("total %d transitions from %d videos" % (tot, len(multi)))

    dev = resolve_device("cuda")

    def batch(idx: np.ndarray, which: str = "train", src=None):
        if src is None:
            src = frames if which == "train" else t_frames
        a = src[idx]
        b = src[idx + 1]
        a = torch.from_numpy(np.asarray(a, np.float32) / 255.0).to(dev)
        b = torch.from_numpy(np.asarray(b, np.float32) / 255.0).to(dev)
        a = a.permute(0, 3, 1, 2)
        b = b.permute(0, 3, 1, 2)
        if args.augment and which == "train":
            # Crop and scale BOTH frames of a pair identically.  Augmenting them
            # independently would destroy the very signal being learned - what moved
            # between the pair - and would show up as "augmentation made it worse".
            side = max(32, int(round(a.shape[-1] * float(np.random.uniform(0.75, 1.0)))))
            x0 = int(np.random.randint(0, a.shape[-1] - side + 1))
            y0 = int(np.random.randint(0, a.shape[-2] - side + 1))
            a = a[:, :, y0:y0 + side, x0:x0 + side]
            b = b[:, :, y0:y0 + side, x0:x0 + side]
        if args.size != a.shape[-1]:
            a = nn.functional.interpolate(a, size=(args.size, args.size), mode="bilinear",
                                          align_corners=False)
            b = nn.functional.interpolate(b, size=(args.size, args.size), mode="bilinear",
                                          align_corners=False)
        if args.augment and which == "train":
            # Same colour transform for the pair, and a per-sample one so the model cannot
            # key on one video's palette (which is what it appears to be doing across
            # videos: folds trained on two videos still fail on the third).
            g = torch.empty(a.shape[0], 3, 1, 1, device=a.device).uniform_(0.8, 1.25)
            o = torch.empty(a.shape[0], 1, 1, 1, device=a.device).uniform_(-0.08, 0.08)
            a = (a * g + o).clamp_(0, 1)
            b = (b * g + o).clamp_(0, 1)
        return a.permute(0, 2, 3, 1), b.permute(0, 2, 3, 1)   # BHWC for ImpalaCNN

    torch.manual_seed(0)
    model = IdmModel(IdmConfig(image_size=args.size,
                               num_actions=DEFAULT_NUM_ACTIONS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        tot, seen = 0.0, 0
        model.train()
        if multi:
            steps = max(1, sum(m[3] for m in multi) // args.batch)
            batches = []
            for _ in range(steps):
                vi = int(rng.choice(len(multi), p=weights))
                _st, fr, mk, n_i = multi[vi]
                batches.append((np.sort(rng.integers(0, n_i, size=args.batch)), fr, mk))
        else:
            rng.shuffle(tr)
            batches = [(np.sort(tr[i:i + args.batch]), frames, masks)
                       for i in range(0, len(tr), args.batch)]
        for idx, fr, mk in batches:
            if len(idx) < 2:
                continue
            a, b = batch(idx, "train", fr)
            y = torch.from_numpy(mk[idx]).to(dev)
            # IdmModel.forward takes (B, H, W, C) and adds the time axis itself;
            # unsqueezing here gave the trunk six dimensions and it refused.
            logits = model(a, b)
            loss = lossf(logits.reshape(-1, logits.shape[-1]), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx); seen += len(idx)
        print("epoch %d train %.4f" % (ep, tot / max(1, seen)), flush=True)

    # ---- evaluation: overall, macro-recall, and the honest "non-noop" accuracy ----
    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(te), 64):
            idx = te[i:i + 64]
            a, b = batch(idx, "test")
            logits = model(a, b)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    pred = np.concatenate(preds)
    y = t_masks[te] if cross else masks[te]
    classes = np.unique(y)
    recalls = {int(c): float((pred[y == c] == c).mean()) for c in classes}
    macro = float(np.mean(list(recalls.values())))
    base = float(np.bincount(y).max() / len(y))
    moving = y != 0
    res = {
        "transitions": int(n), "train": int(len(tr)), "test": int(len(te)),
        "classes_in_test": len(classes), "majority_baseline": base,
        "acc": float((pred == y).mean()), "macro_recall": macro,
        "acc_noop_frames": float((pred[~moving] == y[~moving]).mean()) if (~moving).any() else None,
        "acc_moving_frames": float((pred[moving] == y[moving]).mean()) if moving.any() else 0.0,
        "noop_share_test": float((~moving).mean()),
        "noop_share_pred": float((pred == 0).mean()),
        "per_class_recall": recalls,
    }
    print("\n== %s ==" % ("CROSS-VIDEO: trained on %s, tested on %s"
                           % (args.train_stems or args.stem, t_stem) if cross else
                           "held-out (LAST %.0f%% of the video, temporal)"
                           % (100 * (1 - args.train_frac))))
    print("overall accuracy    %.1f%%   (majority baseline %.1f%%)"
          % (100 * res["acc"], 100 * res["majority_baseline"]))
    print("macro-recall        %.3f   (%d classes seen in the test set)"
          % (macro, len(classes)))
    print("accuracy on MOVING frames (true label != noop) %.1f%%  <- the metric that "
          "a always-noop model scores 0 on" % (100 * res["acc_moving_frames"]))
    print("true noop share %.1f%% | predicted noop share %.1f%%"
          % (100 * res["noop_share_test"], 100 * res["noop_share_pred"]))
    out = Path("runs") / ("idm-keycast-%s.json" % args.stem)
    out.write_text(json.dumps(res, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
