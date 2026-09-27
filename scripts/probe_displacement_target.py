"""Which target should the vision model be asked for?

The pixel-only inverse problem - (frame_t, frame_t+1) -> which key - measured at
chance on every configuration tried, and the reason is not the architecture: the
label is not a function of the pixels in a third of the pairs, and in an interior
room walking moves a small sprite across a static background, so the global
picture change is nearly blind to it.

The engine teacher is now available at training time only, and it offers a
different target: the PLAYER'S OWN DISPLACEMENT.  That is dense, continuous, free
(the agent's own actions produced it) and, unlike a key label, it is exactly what
decides where the character goes.  This script asks whether either target is
learnable from pixels on the same self-play data, plus how much of the action is
decidable from the teacher alone - the chain "vision -> displacement -> action"
can be no better than its second link.

Three measurements on one honest temporal split (never a random one: random
splits leak near-duplicate neighbours, which is how an earlier probe reported a
flattering 91.6%):

  T   teacher displacement -> action class   the ceiling of the chain
  V1  (frame_t, frame_t+1) -> 20 action classes      the old target
  V2  (frame_t, frame_t+1) -> (dpx, dpy)             the proposed target

    uv run python scripts/probe_displacement_target.py runs/selfplay/selfplay-002.npz
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
from ash.loop.bootstrap import _prep  # noqa: E402
from ash.models.idm import IdmConfig, IdmModel  # noqa: E402
from ash.utils.device import resolve_device  # noqa: E402

SIZE = 128
EPOCHS = 8
BATCH = 8
LR = 3e-4          # from-scratch rate, as `pretrain-idm` uses; the in-loop 3e-5
                   # is a fine-tune rate and is not comparable when training a
                   # trunk from random weights.
GRAD_CLIP = 1.0
TRAIN_FRACTION = 0.7


def load(path: str):
    z = np.load(path, allow_pickle=True)
    frames = z["frames"]
    acts = z["acts"].astype(np.int64)
    states = [json.loads(s) for s in z["states"]]
    n = len(acts)
    dpx = np.zeros(n, np.float32)
    dpy = np.zeros(n, np.float32)
    vx = np.zeros(n, np.float32)
    vy = np.zeros(n, np.float32)
    for t in range(n):
        a = (states[t] or {}).get("physics") or {}
        b = (states[t + 1] or {}).get("physics") or {}
        dpx[t] = float(b.get("px") or 0) - float(a.get("px") or 0)
        dpy[t] = float(b.get("py") or 0) - float(a.get("py") or 0)
        vx[t] = float(b.get("vx") or 0)
        vy[t] = float(b.get("vy") or 0)
    return frames, acts, dpx, dpy, vx, vy


def teacher_ceiling(acts, dpx, dpy, vx, vy, train, test) -> dict:
    """How much of the action is readable from the teacher's own numbers?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.stack([dpx, dpy, vx, vy], axis=1)
    sc = StandardScaler().fit(x[train])
    clf = LogisticRegression(max_iter=400)
    clf.fit(sc.transform(x[train]), acts[train])
    pred = clf.predict(sc.transform(x[test]))
    classes = np.unique(acts[test])
    recalls = [float((pred[acts[test] == c] == c).mean()) for c in classes if (acts[test] == c).any()]
    base = float(np.bincount(acts[test]).max() / len(acts[test]))
    return {"acc": float((pred == acts[test]).mean()), "chance": 1.0 / len(classes),
            "baseline": base, "macro_recall": float(np.mean(recalls)), "n_classes": len(classes)}


def vision_targets(dev, frames, acts, dpx, dpy, train, test, tag: str) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)
    model = IdmModel(IdmConfig(image_size=SIZE, num_actions=len(ActionSpace.minimal()))).to(dev)
    # 生产特征是 [ea, eb, eb - ea]（trunk 输出），不是 IdmModel.forward 的动作 logits：
    # 后者已经过了一个头，再套一层就等于在 logits 上学习。
    n_feat = model.config.embed_dim * 3
    if tag == "action":
        head = nn.Linear(n_feat, len(ActionSpace.minimal())).to(dev)
        loss_fn: nn.Module = nn.CrossEntropyLoss()
        target = torch.from_numpy(acts).to(dev)
    else:
        head = nn.Linear(n_feat, 2).to(dev)
        loss_fn = nn.MSELoss()
        mu = np.stack([dpx, dpy], axis=1)[train].mean(axis=0)
        sd = np.stack([dpx, dpy], axis=1)[train].std(axis=0) + 1e-6
        target = torch.from_numpy(((np.stack([dpx, dpy], axis=1) - mu) / sd).astype(np.float32)).to(dev)
        model.register_buffer("_mu", torch.from_numpy(mu.astype(np.float32)))
        model.register_buffer("_sd", torch.from_numpy(sd.astype(np.float32)))

    params = list(model.trunk.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    order = np.arange(len(train))
    rng = np.random.default_rng(0)

    def forward(idx):
        a = torch.from_numpy(_prep(frames[idx], SIZE)).to(dev)
        b = torch.from_numpy(_prep(frames[idx + 1], SIZE)).to(dev)
        ea = model.trunk(a.unsqueeze(1)).squeeze(1)
        eb = model.trunk(b.unsqueeze(1)).squeeze(1)
        return head(torch.cat([ea, eb, eb - ea], dim=1))

    for epoch in range(EPOCHS):
        rng.shuffle(order)
        total, seen = 0.0, 0
        model.train()
        for i in range(0, len(order), BATCH):
            idx = train[order[i : i + BATCH]]
            loss = loss_fn(forward(idx), target[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            total += float(loss) * len(idx); seen += len(idx)
        print("    [%s] epoch %d train %.4f" % (tag, epoch, total / max(1, seen)), flush=True)

    model.eval()
    with torch.inference_mode():
        out = torch.cat([forward(test[i : i + 64]) for i in range(0, len(test), 64)])

    if tag == "action":
        pred = out.argmax(dim=1).cpu().numpy()
        y = acts[test]
        classes = np.unique(y)
        recalls = [float((pred[y == c] == c).mean()) for c in classes if (y == c).any()]
        return {"acc": float((pred == y).mean()), "macro_recall": float(np.mean(recalls)),
                "baseline": float(np.bincount(y).max() / len(y)), "chance": 1.0 / len(classes),
                "pred_noop": float((pred == 0).mean()), "true_noop": float((y == 0).mean())}

    pred = (out.cpu().numpy() * sd + mu)
    true = np.stack([dpx, dpy], axis=1)[test]
    out_d: dict[str, object] = {}
    for j, axis in enumerate(("dpx", "dpy")):
        mse = float(((pred[:, j] - true[:, j]) ** 2).mean())
        zero = float(((true[:, j].mean() - true[:, j]) ** 2).mean())
        r = float(np.corrcoef(pred[:, j], true[:, j])[0, 1]) if true[:, j].std() > 0 else float("nan")
        out_d[axis] = {"mse": mse, "baseline_mse": zero,
                       "skill": 1.0 - mse / zero if zero > 0 else float("nan"),
                       "corr": r, "std_true": float(true[:, j].std()),
                       "std_pred": float(pred[:, j].std())}
    return out_d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    args = ap.parse_args()

    dev = resolve_device("cuda")
    frames, acts, dpx, dpy, vx, vy = load(args.path)
    n = len(acts)
    cut = int(args.train_fraction * n)
    train = np.arange(cut)
    test = np.arange(cut, n)
    print("数据 %s：%d 个转移，时间序切分（前 %d 训练 / 后 %d 测试）"
          % (Path(args.path).name, n, len(train), len(test)))
    print("位移真值：dpx 中位 %.1f（std %.1f）| dpy 中位 %.1f（std %.1f）| 真 noop 比例 %.1f%%"
          % (np.median(dpx), dpx.std(), np.median(dpy), dpy.std(),
             100 * (acts == 0).mean()))

    print("\n=== T：只用老师的数字（Δpx, Δpy, vx, vy）判动作 ===")
    t = teacher_ceiling(acts, dpx, dpy, vx, vy, train, test)
    print("  准确率 %.1f%% | 基线 %.1f%% | 随机(按类数) %.1f%% | macro-recall %.3f | 类数 %d"
          % (100 * t["acc"], 100 * t["baseline"], 100 * t["chance"],
             t["macro_recall"], t["n_classes"]))

    print("\n=== V1：像素 → 20 类动作（旧目标）===")
    v1 = vision_targets(dev, frames, acts, dpx, dpy, train, test, "action")

    print("\n=== V2：像素 → 玩家位移 (Δpx, Δpy)（新目标）===")
    v2 = vision_targets(dev, frames, acts, dpx, dpy, train, test, "displacement")

    print("\n=== 汇总 ===")
    print("T  老师数字→动作      %.1f%%（基线 %.1f%%）| macro-recall %.3f"
          % (100 * t["acc"], 100 * t["baseline"], t["macro_recall"]))
    print("V1 像素→动作          %.1f%%（基线 %.1f%%）| macro-recall %.3f | 预测noop %.1f%%（真 %.1f%%）"
          % (100 * v1["acc"], 100 * v1["baseline"], v1["macro_recall"],
             100 * v1["pred_noop"], 100 * v1["true_noop"]))
    for axis in ("dpx", "dpy"):
        d = v2[axis]
        print("V2 像素→%-3s          MSE %.1f（零预测基线 %.1f）| skill %+0.3f | corr %+0.3f"
              " | 预测std %.1f vs 真std %.1f"
              % (axis, d["mse"], d["baseline_mse"], d["skill"], d["corr"],
                 d["std_pred"], d["std_true"]))
    print("\n判读：skill<=0 或 corr≈0 ⇒ 连「画面→位移」都学不到；"
          "T 高而 V2 低 ⇒ 瓶颈在视觉侧；T 本身就低 ⇒ 老师的数字不足以定动作，链条第二环就断了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
