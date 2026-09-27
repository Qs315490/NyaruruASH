"""Does a difference branch fix the IDM's cross-scene failure?

The probe that motivated this: the motion signal cannot travel through
`[ea, eb, eb - ea]` built from global-average-pooled features. Held out, that
structure scored 49.0% on "adjacent vs far apart" while the bare pixel difference
scored 79.2% and the same trunk fed the DIFFERENCE IMAGE scored 85.7%.

The IDM's action head uses the pooled-difference structure, so this adds the
difference branch it lacks - trunk(a), trunk(b), trunk(|a - b|) - and re-runs the
cross-scene criterion that the current model fails:

    current IDM, train 3 house recordings, hold out human-001
        accuracy 45.5%   majority baseline 50.4%

Pass means beating the baseline by a real margin, not by a lucky seed.

    uv run python scripts/probe_diff_branch_idm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.actions.space import ActionSpace  # noqa: E402
from ash.loop.bootstrap import _prep  # noqa: E402
from ash.models.idm import IdmConfig, IdmModel  # noqa: E402
from ash.train.demo_mapping import load_demo, map_legacy_masks  # noqa: E402
from ash.utils.device import resolve_device  # noqa: E402

SIZE = 128
NAMES = ["house-001", "house-002", "house-003", "human-001"]
HOLDOUT = "human-001"
EPOCHS = 8
LR = 3e-5
BATCH = 8


def main() -> int:
    dev = resolve_device("cuda")
    rng = np.random.default_rng(0)
    space = ActionSpace.minimal()

    data = {}
    for name in NAMES:
        demo = load_demo(f"data/recordings/{name}.npz")
        obs = np.asarray(demo["observations"])
        index, _ = map_legacy_masks(np.asarray(demo["control_masks"]), space)
        # 存成 128 的 uint8，避免把 256 的原图整段留在内存里
        small = np.stack([cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                          for f in obs])
        data[name] = (small, index[1:].astype(np.int64))
        print("%-11s %5d 帧" % (name, len(small)))

    def pairs(name: str):
        frames, lab = data[name]
        return frames[:-1], frames[1:], lab

    a_tr, b_tr, y_tr = [], [], []
    for name in NAMES:
        if name == HOLDOUT:
            continue
        x, z, lab = pairs(name)
        a_tr.append(x); b_tr.append(z); y_tr.append(lab)
    a_tr = np.concatenate(a_tr); b_tr = np.concatenate(b_tr)
    y_tr = torch.from_numpy(np.concatenate(y_tr))
    a_te, b_te, y_te = pairs(HOLDOUT)
    y_te_t = torch.from_numpy(y_te)
    base = max((y_te == 0).mean(), (y_te != 0).mean())
    print("训练 %d 对 | 留出 %s %d 对 | 真 noop %.1f%% | 多数类基线 %.1f%%"
          % (len(y_tr), HOLDOUT, len(y_te), 100 * (y_te == 0).mean(), 100 * base))

    num_actions = len(space)

    def run_arm(use_diff: bool, seed: int = 0) -> dict:
        """Train one arm and score it.  `use_diff=False` is the production head.

        Both arms share the split, the seed, the optimizer, the epochs and the
        best-epoch selection, so the only difference between their numbers is the
        head.  That matters: the "45.5%" this probe used to quote for the current
        model was a static string from a differently configured run, which is not
        a comparison.
        """
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = IdmModel(IdmConfig(image_size=SIZE, num_actions=num_actions)).to(dev)
        # The difference branch: the trunk also sees the per-pixel difference
        # image, because pooling each frame first and then subtracting (which is
        # what the production head does) destroys where the motion was.
        # Production features are [ea, eb, eb - ea] (3 * embed_dim).  The
        # difference arm appends one more trunk output, ed = trunk(|a - b|).
        n_feat = 4 if use_diff else 3
        head = nn.Sequential(
            nn.Linear(model.config.embed_dim * n_feat, model.config.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model.config.embed_dim, num_actions),
        ).to(dev)
        opt = torch.optim.AdamW(list(model.trunk.parameters()) + list(head.parameters()), lr=LR)
        ce = nn.CrossEntropyLoss()

        def features(x, z):
            ta = torch.from_numpy(_prep(x, SIZE)).to(dev)
            tb = torch.from_numpy(_prep(z, SIZE)).to(dev)
            ea = model.trunk(ta.unsqueeze(1)).squeeze(1)
            eb = model.trunk(tb.unsqueeze(1)).squeeze(1)
            parts = [ea, eb, eb - ea]
            if use_diff:
                parts.append(model.trunk(torch.abs(ta - tb).unsqueeze(1)).squeeze(1))
            return torch.cat(parts, dim=1)

        params = list(model.trunk.parameters()) + list(head.parameters())
        order = np.arange(len(a_tr))
        best_state, best_head, best_loss = None, None, float("inf")
        for epoch in range(EPOCHS):
            rng.shuffle(order)
            total, seen = 0.0, 0
            for i in range(0, len(order), BATCH):
                take = order[i:i + BATCH]
                loss = ce(head(features(a_tr[take], b_tr[take])), y_tr[take].to(dev))
                if not torch.isfinite(loss):
                    print("  [%s] epoch %d: 非有限 loss，停止本轮" % (arm, epoch))
                    break
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                total += float(loss) * len(take); seen += len(take)
            mean_loss = total / max(1, seen)
            if np.isfinite(mean_loss) and mean_loss < best_loss:
                best_loss = mean_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}
            print("  [%s] epoch %d: loss %.4f%s"
                  % (arm, epoch, mean_loss, "  (最优)" if best_loss == mean_loss else ""),
                  flush=True)
        if best_state is not None:
            model.load_state_dict(best_state); head.load_state_dict(best_head)

        with torch.no_grad():
            preds = [head(features(a_te[i:i + 64], b_te[i:i + 64])).argmax(-1).cpu().numpy()
                     for i in range(0, len(a_te), 64)]
        pred = np.concatenate(preds)
        moving = y_te != 0
        acc, mov = float((pred == y_te).mean()), float((pred[moving] == y_te[moving]).mean())
        recalls = [float(((pred == c) & (y_te == c)).sum() / max(1, (y_te == c).sum()))
                   for c in np.unique(y_te)]
        print("  [%s] 留出 %.1f%%（基线 %.1f%%）| 运动帧上 %.1f%% | 预测运动 %.1f%% | macro-recall %.3f"
              % (arm, 100 * acc, 100 * base, 100 * mov, 100 * (pred != 0).mean(),
                 float(np.mean(recalls))), flush=True)
        return {"acc": acc, "moving": mov, "recall": float(np.mean(recalls))}

    print("\n两侧对照（同一份数据、同一个种子、同一套超参 lr=%g epochs=%d）:" % (LR, EPOCHS))
    out = {}
    for arm, use_diff in (("生产头 [ea,eb,eb-ea]", False), ("差异分支 +|a-b|", True)):
        out[arm] = run_arm(use_diff)

    print("\n=== 结论 ===")
    for arm, r in out.items():
        print("%-22s 留出 %.1f%% | 运动帧上 %.1f%% | macro-recall %.3f"
              % (arm, 100 * r["acc"], 100 * r["moving"], r["recall"]))
    gain = 100 * (out["差异分支 +|a-b|"]["acc"] - out["生产头 [ea,eb,eb-ea]"]["acc"])
    print("差异分支相对生产头：%+.1f 个百分点（基线 %.1f%%）" % (gain, 100 * base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
