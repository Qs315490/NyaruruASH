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


def pairs_of(path: str):
    """(a, b, acts, dpx, dpy, vx, vy) for one session.

    Pairs are built INSIDE a session and only then concatenated, so no phantom
    transition is created across the seam between two recordings.
    """
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
    return frames[:-1], frames[1:], acts, dpx, dpy, vx, vy


def teacher_ceiling(y_tr, x_tr, y_te, x_te) -> dict:
    """How much of the action is readable from the teacher's own numbers?

    Takes the two sides explicitly.  An earlier version indexed a single array
    with the train/test index sets while the arrays had already been split, which
    raised IndexError the moment the split stopped being one session.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(x_tr)
    clf = LogisticRegression(max_iter=400)
    clf.fit(sc.transform(x_tr), y_tr)
    pred = clf.predict(sc.transform(x_te))
    classes = np.unique(y_te)
    recalls = [float((pred[y_te == c] == c).mean()) for c in classes if (y_te == c).any()]
    base = float(np.bincount(y_te).max() / len(y_te))
    return {"acc": float((pred == y_te).mean()), "chance": 1.0 / len(classes),
            "baseline": base, "macro_recall": float(np.mean(recalls)), "n_classes": len(classes)}


def vision_targets(dev, tr, te, tag: str) -> dict:
    """Train on `tr` and score on `te`, each a tuple of arrays.

    Both sides are passed explicitly because an earlier version took the target
    arrays from ONE session and indexed them with the other session's indices:
    with train and test the same length it did not raise, it trained on the test
    targets, and the leak showed up as two different splits reporting identical
    numbers to three decimals.  A split that cannot fail loudly has to be
    impossible to write by accident, so the data now travels as tuples.
    """
    a_tr, b_tr, y_tr, dpx_tr, dpy_tr = tr
    a_te, b_te, y_te, dpx_te, dpy_te = te
    torch.manual_seed(0)
    np.random.seed(0)
    model = IdmModel(IdmConfig(image_size=SIZE, num_actions=len(ActionSpace.minimal()))).to(dev)
    n_feat = model.config.embed_dim * 3
    if tag == "action":
        head = nn.Linear(n_feat, len(ActionSpace.minimal())).to(dev)
        loss_fn: nn.Module = nn.CrossEntropyLoss()
        target = torch.from_numpy(y_tr).to(dev)
    else:
        head = nn.Linear(n_feat, 2).to(dev)
        loss_fn = nn.MSELoss()
        stacked = np.stack([dpx_tr, dpy_tr], axis=1)
        mu, sd = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6
        target = torch.from_numpy(((stacked - mu) / sd).astype(np.float32)).to(dev)

    params = list(model.trunk.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    idx_all = np.arange(len(y_tr))
    rng = np.random.default_rng(0)

    def forward(x, y, idx):
        xb = torch.from_numpy(_prep(x[idx], SIZE)).to(dev)
        yb = torch.from_numpy(_prep(y[idx], SIZE)).to(dev)
        ea = model.trunk(xb.unsqueeze(1)).squeeze(1)
        eb = model.trunk(yb.unsqueeze(1)).squeeze(1)
        return head(torch.cat([ea, eb, eb - ea], dim=1))

    for epoch in range(EPOCHS):
        rng.shuffle(idx_all)
        total, seen = 0.0, 0
        model.train()
        for i in range(0, len(idx_all), BATCH):
            idx = idx_all[i : i + BATCH]
            loss = loss_fn(forward(a_tr, b_tr, idx), target[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            total += float(loss) * len(idx); seen += len(idx)
        print("    [%s] epoch %d train %.4f" % (tag, epoch, total / max(1, seen)), flush=True)

    model.eval()
    test_idx = np.arange(len(y_te))
    with torch.inference_mode():
        out = torch.cat([forward(a_te, b_te, test_idx[i : i + 64])
                         for i in range(0, len(test_idx), 64)])

    if tag == "action":
        pred = out.argmax(dim=1).cpu().numpy()
        classes = np.unique(y_te)
        recalls = [float((pred[y_te == c] == c).mean()) for c in classes if (y_te == c).any()]
        return {"acc": float((pred == y_te).mean()), "macro_recall": float(np.mean(recalls)),
                "baseline": float(np.bincount(y_te).max() / len(y_te)),
                "chance": 1.0 / len(classes), "pred_noop": float((pred == 0).mean()),
                "true_noop": float((y_te == 0).mean())}

    pred = out.cpu().numpy() * sd + mu
    true = np.stack([dpx_te, dpy_te], axis=1)
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
    ap.add_argument("--train", nargs="+", required=True, help="训练会话 npz（可多个）")
    ap.add_argument("--test", required=True, help="留出会话 npz")
    args = ap.parse_args()

    dev = resolve_device("cuda")
    cols = list(zip(*[pairs_of(q) for q in args.train]))
    a_tr = np.concatenate(cols[0]); b_tr = np.concatenate(cols[1])
    acts_tr = np.concatenate(cols[2]); dpx_tr = np.concatenate(cols[3])
    dpy_tr = np.concatenate(cols[4]); vx_tr = np.concatenate(cols[5])
    vy_tr = np.concatenate(cols[6])

    a_te, b_te, acts_te, dpx_te, dpy_te, vx_te, vy_te = pairs_of(args.test)
    print("训练 %d 对（%d 个会话）| 留出 %s %d 对"
          % (len(acts_tr), len(args.train), Path(args.test).name, len(acts_te)))
    print("位移真值：dpx std %.1f | dpy std %.1f | 真 noop %.1f%%"
          % (dpx_te.std(), dpy_te.std(), 100 * (acts_te == 0).mean()))

    def feats(dpx, dpy, vx, vy):
        return np.stack([dpx, dpy, vx, vy], axis=1)

    print("\n=== T：只用老师的数字（Δpx, Δpy, vx, vy）判动作 ===")
    t = teacher_ceiling(acts_tr, feats(dpx_tr, dpy_tr, vx_tr, vy_tr),
                        acts_te, feats(dpx_te, dpy_te, vx_te, vy_te))
    print("  准确率 %.1f%% | 基线 %.1f%% | 随机(按类数) %.1f%% | macro-recall %.3f | 类数 %d"
          % (100 * t["acc"], 100 * t["baseline"], 100 * t["chance"],
             t["macro_recall"], t["n_classes"]))

    tr_tuple = (a_tr, b_tr, acts_tr, dpx_tr, dpy_tr)
    te_tuple = (a_te, b_te, acts_te, dpx_te, dpy_te)
    print("\n=== V1：像素 → 20 类动作 ===")
    v1 = vision_targets(dev, tr_tuple, te_tuple, "action")

    print("\n=== V2：像素 → 玩家位移 (Δpx, Δpy) ===")
    v2 = vision_targets(dev, tr_tuple, te_tuple, "displacement")

    print("\n=== 汇总（留出会话 = %s）===" % Path(args.test).name)
    print("T  老师数字→动作   %.1f%%（基线 %.1f%%）| macro-recall %.3f"
          % (100 * t["acc"], 100 * t["baseline"], t["macro_recall"]))
    print("V1 像素→动作       %.1f%%（基线 %.1f%%）| macro-recall %.3f"
          % (100 * v1["acc"], 100 * v1["baseline"], v1["macro_recall"]))
    for axis in ("dpx", "dpy"):
        d = v2[axis]
        print("V2 像素→%-3s       skill %+0.3f | corr %+0.3f | 预测std %.1f vs 真std %.1f"
              % (axis, d["skill"], d["corr"], d["std_pred"], d["std_true"]))
    print("\n判读：skill 明显为正且 corr 高 ⇒ 位移目标在该留出集上可学。"
          "\n     与域内（corr 0.92）比较即可看出跨区域损失了多少。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
