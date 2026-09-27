"""The labeler works in the room it was trained in.  What does it say on video?

This is the step the whole project turns on, and it has never been measured with
a labeler that works: the IDM is trained on the agent's own self-play transitions
(no human data anywhere), a held-out session says how good it is, and then it is
pointed at the reference videos - which have no labels at all, so the question is
not "is it right" but "what is it saying".

Three outcomes and what they mean:

  degenerate on video   one class for every pair, or a bias-dominated argmax.
                        Then pi would be taught a constant, which is the failure
                        this project spent its history on.
  plausible distribution  a spread of classes, noop share below the in-domain
                        value, and an argmax that moves with the input.  Then the
                        pseudo-labels are usable and pi can be trained on them.
  nothing in between    anything else needs reading before it means anything.

The in-domain number is printed first on purpose: a cross-domain verdict from a
labeler that cannot label its own data would be worthless.

    uv run python scripts/probe_selfplay_to_video.py
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

from ash.actions.space import ActionSpace, buttons_from_mask  # noqa: E402
from ash.data.corpus_crop import CroppedFrames, load_crops  # noqa: E402
from ash.data.effect import is_effective  # noqa: E402
from ash.loop.bootstrap import _prep, action_names, logit_diagnosis  # noqa: E402
from ash.models.idm import IdmConfig, IdmModel  # noqa: E402
from ash.utils.device import resolve_device  # noqa: E402

SIZE = 128
EPOCHS = 8
BATCH = 8
LR = 3e-4
GRAD_CLIP = 1.0


def load_session(path: str):
    z = np.load(path, allow_pickle=True)
    return (np.asarray(z["frames"]), np.asarray(z["acts"], dtype=np.int64),
            [json.loads(s) for s in z["states"]])


def effective_pairs(path: str, masks):
    frames, acts, states = load_session(path)
    keep = np.asarray([is_effective(states[t], states[t + 1], masks[int(acts[t])])["effective"] is True
                       for t in range(len(acts))], dtype=bool)
    return frames[:-1][keep], frames[1:][keep], acts[keep]


def train(dev, train_paths, masks, epochs: int) -> IdmModel:
    a, b, y = [], [], []
    for path in train_paths:
        pa, pb, py = effective_pairs(path, masks)
        a.append(pa); b.append(pb); y.append(py)
        print("  %-28s 有效对 %d" % (Path(path).name, len(py)))
    a = np.concatenate(a); b = np.concatenate(b); y = np.concatenate(y)
    print("训练共 %d 个有效对" % len(y))

    torch.manual_seed(0)
    model = IdmModel(IdmConfig(image_size=SIZE, num_actions=len(ActionSpace.minimal()))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    order = np.arange(len(y))
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        rng.shuffle(order)
        total, seen = 0.0, 0
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            x = torch.from_numpy(_prep(a[idx], SIZE)).to(dev)
            z = torch.from_numpy(_prep(b[idx], SIZE)).to(dev)
            t = torch.from_numpy(y[idx]).to(dev)
            loss = ce(model(x, z), t)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += float(loss) * len(idx); seen += len(idx)
        print("  epoch %d train %.4f" % (epoch, total / max(1, seen)), flush=True)
    model.eval()
    return model


def score(model, dev, a, b, y, tag: str) -> dict:
    logits = []
    with torch.inference_mode():
        for i in range(0, len(a), 64):
            x = torch.from_numpy(_prep(a[i : i + 64], SIZE)).to(dev)
            z = torch.from_numpy(_prep(b[i : i + 64], SIZE)).to(dev)
            logits.append(model(x, z).cpu().numpy())
    L = np.concatenate(logits)
    pred = L.argmax(1)
    classes = np.unique(y)
    recalls = [float((pred[y == c] == c).mean()) for c in classes if (y == c).any()]
    names = action_names(len(ActionSpace.minimal()))
    counts = np.bincount(pred, minlength=len(names))
    out = {
        "acc": float((pred == y).mean()),
        "baseline": float(np.bincount(y, minlength=len(names)).max() / len(y)),
        "macro_recall": float(np.mean(recalls)),
        "noop_share": float((pred == 0).mean()),
        "top_share": float(counts.max() / max(1, counts.sum())),
        "classes_used": int((counts > 0).sum()),
        "top3": [(names[i], int(counts[i])) for i in np.argsort(-counts)[:3]],
        "diag": logit_diagnosis(L),
    }
    print("  [%s] n=%d | 准确率 %.1f%%（基线 %.1f%%）| macro-recall %.3f | "
          "预测noop %.1f%% | 用到 %d 类 | top: %s"
          % (tag, len(y), 100 * out["acc"], 100 * out["baseline"], out["macro_recall"],
             100 * out["noop_share"], out["classes_used"],
             ", ".join("%s %.0f%%" % (n, 100 * c / len(y)) for n, c in out["top3"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=[
        "runs/selfplay/selfplay-005.npz", "runs/selfplay/selfplay-006.npz",
        "runs/selfplay/selfplay-007.npz", "runs/selfplay/selfplay-008.npz"])
    ap.add_argument("--holdout", default="runs/selfplay/selfplay-002.npz")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--crops", default="data/corpus-crops.json")
    ap.add_argument("--frames", type=int, default=400, help="每个语料视频抽样多少帧")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    dev = resolve_device("cuda")
    masks = ActionSpace.minimal().masks
    print("=== 训练 IDM（只用 agent 自己的转移）===")
    model = train(dev, args.train, masks, args.epochs)

    print("\n=== 域内：留出的自博弈会话 ===")
    a, b, y = effective_pairs(args.holdout, masks)
    score(model, dev, a, b, y, "自博弈留出")

    print("\n=== 跨域：参考视频（无标签，只看它说了什么）===")
    crops = load_crops(args.crops)
    results = {}
    for path in sorted(Path(args.corpus).glob("*.npy")):
        stem = path.stem
        arr = np.load(path, mmap_mode="r")
        rect = crops.get(stem) if isinstance(crops, dict) else None
        frames = CroppedFrames(arr, rect) if rect else arr
        take = np.linspace(0, len(frames) - 2, min(args.frames, len(frames) - 1)).astype(int)
        import cv2

        small = np.stack([cv2.resize(frames[i], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                          for i in take])
        nxt = np.stack([cv2.resize(frames[i + 1], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                        for i in take])
        # No labels exist for these; score() only needs a y to compute accuracy,
        # so pass a placeholder and read the distribution fields.
        results[stem] = score(model, dev, small, nxt, np.zeros(len(take), np.int64), stem)

    print("\n=== 汇总：语料上的伪标签 ===")
    print("%-18s %8s %9s %9s %10s" % ("视频", "预测noop", "最大类占比", "用到类数", "bias/temporal"))
    for stem, r in results.items():
        print("%-18s %7.1f%% %8.1f%% %9d %10s"
              % (stem, 100 * r["noop_share"], 100 * r["top_share"], r["classes_used"],
                 ("%.2f" % r["diag"]["bias_over_temporal"]) if r["diag"] else "?"))
    print("\n判读：预测noop 长期贴近 100% 或 bias/temporal 远大于 1 ⇒ 这个标签器在视频上是退化的，"
          "\n      π 拿去训只会被教成常量；若分布有铺开，才值得把 π 接上去。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
