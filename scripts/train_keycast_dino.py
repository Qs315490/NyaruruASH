"""IDM on frozen DINOv2 features - a pretrained backbone instead of a small ImpalaCNN.

Three levers have been ruled out already: more data, augmentation, and the objective's
form.  What remains is the representation, and this tests the cheapest version of that:
keep the corpus exactly as it is, but replace the randomly initialised trunk with frozen
DINOv2 ViT-S/14 features (the same pretrained model the memory K already uses).

The trunk is frozen, so the features are computed once per video and cached; the trainable
part is a small head over [ea, eb, eb - ea], the same concatenation the CNN IDM used.  That
makes each fold seconds rather than minutes, so this is cheap to try and cheap to trust.

The metric is macro-F1 over keys, same as the multi-label run, because it is the number an
all-empty prediction cannot fake.

    uv run python scripts/train_keycast_dino.py --train-stems A,B --test-stem C
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

from ash.memory.embeddings import FrameEmbedder  # noqa: E402
from ash.utils.device import resolve_device  # noqa: E402


def features(stem: str, emb: FrameEmbedder) -> np.ndarray:
    """DINOv2 features for one video's corpus frames, cached on disk."""
    out = Path("runs") / ("dino-%s.npy" % stem)
    fr = np.load(Path("data/corpus/%s.npy" % stem), mmap_mode="r")
    if out.exists():
        E = np.load(out)
        if len(E) == len(fr):
            print("cached  %-14s %s %s" % (stem, E.shape, out))
            return E
        print("stale cache for %s (%d vs %d frames), recomputing" % (stem, len(E), len(fr)))
    chunks = []
    for i in range(0, len(fr), 512):
        chunks.append(emb.embed(np.asarray(fr[i:i + 512]), batch_size=64))
    E = np.concatenate(chunks).astype(np.float32)
    np.save(out, E)
    print("embedded %-14s %s -> %s" % (stem, E.shape, out))
    return E


def held_of(stem: str):
    L = np.load(Path("runs/keycast-labels-%s.npz" % stem), allow_pickle=True)
    return L["held"].astype(np.float32), [str(n) for n in L["names"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stems", required=True)
    ap.add_argument("--test-stem", required=True)
    ap.add_argument("--tag", default="dino")
    ap.add_argument("--feat-mode", default="cat", choices=["cat", "absdiff", "both"],
                    help="cat=[ea,eb,eb-ea]  absdiff=[ea,eb,|ea-eb|]  both=all four")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    dev = resolve_device("cuda")
    emb = FrameEmbedder()
    train_stems = [x.strip() for x in args.train_stems.split(",") if x.strip()]
    keys = []
    data = []
    for st in train_stems:
        E = features(st, emb)
        held, names = held_of(st)
        n = min(len(E) - 1, len(held) - 1)
        data.append((E, held, names, n))
        keys += names
        print("train %-14s %d transitions, %d keys" % (st, n, len(names)))
    KEY_ORDER = sorted(set(keys))
    kidx = {k: i for i, k in enumerate(KEY_ORDER)}
    print("keys (%d): %s" % (len(KEY_ORDER), KEY_ORDER))

    def targets(held, names):
        y = np.zeros((len(held), len(KEY_ORDER)), np.float32)
        for j, nm in enumerate(names):
            if nm in kidx:
                y[:, kidx[nm]] = held[:, j]
        return y

    train = [(E, targets(h, nm), n) for E, h, nm, n in data]
    tot = sum(t[2] for t in train)
    weights = np.array([t[2] / tot for t in train])
    tE = features(args.test_stem, emb)
    t_held, t_names = held_of(args.test_stem)
    t_n = min(len(tE) - 1, len(t_held) - 1)
    tY = targets(t_held, t_names)
    print("test  %-14s %d transitions" % (args.test_stem, t_n))

    D = tE.shape[1]
    torch.manual_seed(0)

    # The task shape, not just the backbone: the documented positive fact in this repo is
    # that |a-b| fed into the trunk reaches 85.7% within one pipeline, while [ea, eb, eb-ea]
    # (a pooled subtraction) reached 49.0%.  Note |ea-eb| is not the same thing as |a-b| of
    # the pixels, but it is the cheap version of it and it flips the sign information into a
    # magnitude, which is what "something moved" looks like.
    n_in = {"cat": 3, "absdiff": 3, "both": 4}[args.feat_mode]

    def head():
        return nn.Sequential(
            nn.LayerNorm(n_in * D), nn.Linear(n_in * D, 512), nn.GELU(),
            nn.Linear(512, len(KEY_ORDER)),
        ).to(dev)

    def feats(E, held, y, idx):
        ea = torch.from_numpy(E[idx]).to(dev)
        eb = torch.from_numpy(E[idx + 1]).to(dev)
        if args.feat_mode == "cat":
            x = torch.cat([ea, eb, eb - ea], dim=1)
        elif args.feat_mode == "absdiff":
            x = torch.cat([ea, eb, (ea - eb).abs()], dim=1)
        else:
            x = torch.cat([ea, eb, eb - ea, (ea - eb).abs()], dim=1)
        return x, torch.from_numpy(y[idx]).to(dev)

    model = head()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        model.train()
        steps = max(1, tot // args.batch)
        tl = 0.0
        for _ in range(steps):
            vi = int(rng.choice(len(train), p=weights))
            E, y, n = train[vi]
            idx = np.sort(rng.integers(0, n, size=args.batch))
            x, yy = feats(E, None, y, idx)
            loss = lossf(model(x), yy)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += float(loss.detach())
        if ep % 5 == 0 or ep == args.epochs - 1:
            print("epoch %2d train %.4f" % (ep, tl / steps), flush=True)

    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, t_n, 1024):
            idx = np.arange(i, min(i + 1024, t_n))
            x, _ = feats(tE, None, tY, idx)
            preds.append((model(x) > 0).float().cpu().numpy())
    P = np.concatenate(preds)
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
    res = {"backbone": "dinov2_vits14 (frozen)", "feat_mode": args.feat_mode,
           "train": train_stems,
           "test": args.test_stem, "macro_f1": round(macro, 3),
           "bit_accuracy": round(float((P == Y).mean()), 4),
           "accuracy_on_active_bits": round(float((P[ones] == Y[ones]).mean()), 4) if ones.any() else None,
           "exact_set_match": round(float((P == Y).all(axis=1).mean()), 4),
           "pred_positive_rate": round(float(P.mean()), 4),
           "true_positive_rate": round(float(Y.mean()), 4), "per_key": per}
    print("\n== FROZEN DINOv2 [%s]: train %s -> test %s =="
          % (args.feat_mode, train_stems, args.test_stem))
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
    out = Path("runs") / ("idm-dino-%s-%s-%s.json" % (args.test_stem, args.feat_mode, args.tag))
    out.write_text(json.dumps(res, indent=1))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
