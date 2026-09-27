"""How much of the labelled data can a pixel-only model even learn from?

The user's correction, and it reframes the whole failure: frames where nothing
moves are NOT junk.  A human correctly presses nothing while a cutscene types
itself out, while a moving platform carries them, and they press ``left`` into a
wall, which also produces no motion.  So "the screen did not change" is
compatible with both "no key" and "a key whose effect the world cancelled", and
the inverse problem (pixels -> key) stops being a function.

That is a stronger explanation than "too little data" or "wrong architecture",
and it predicts two things this script measures on the existing recordings:

  * a large share of still pairs carry a MOVEMENT label -> those pairs teach
    "static screen means left", which contradicts the noop pairs beside them;
  * a large share of moving pairs carry a NOOP label -> world motion under a
    cutscene or a platform, which teaches the opposite.

The irreducible part is their sum: pairs whose label the pixels cannot
determine.  Any accuracy ceiling follows from it, and no model can pass it.

    uv run python scripts/probe_label_identifiability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.actions.space import ActionSpace  # noqa: E402
from ash.memory.embeddings import DEFAULT_IMAGE_SIZE  # noqa: E402
from ash.train.demo_mapping import load_demo, map_legacy_masks  # noqa: E402

SIZE = 128
NOOP = "noop"


def action_names() -> list[str]:
    from ash.actions.space import buttons_from_mask

    masks = ActionSpace.minimal().masks
    return [", ".join(buttons_from_mask(m)) or NOOP for m in masks]


def load_sources() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{source: (frames @SIZE, mask label at t for each transition)}."""
    space = ActionSpace.minimal()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    demo = Path("data/idm-human")
    obs = np.load(demo / "observations.npy", mmap_mode="r")
    eps = np.load(demo / "episode_ids.npy", mmap_mode="r")
    masks = np.load(demo / "control_masks.npy", mmap_mode="r")
    for e in np.unique(eps):
        rows = np.flatnonzero(eps == e)
        index, _ = map_legacy_masks(np.asarray(masks[rows]), space)
        out[f"human:ep{int(e):02d}"] = (np.asarray(obs[rows]), index)
    for name in ("house-001", "house-002", "house-003"):
        z = np.load(f"data/recordings/{name}.npz")
        o = z["observations"]
        small = np.stack([cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for f in o])
        index, _ = map_legacy_masks(np.asarray(z["control_masks"]), space)
        out[name] = (small, index)
    return out


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative)."""
    r = scores.argsort().argsort().astype(np.float64)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return float((r[positive].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def main() -> int:
    names = action_names()
    sources = load_sources()
    motion, label, source = [], [], []
    for name, (frames, lab) in sources.items():
        # One transition per executed step, and the label is the key state at t,
        # exactly as the IDM sees them.
        d = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
        motion.append(d.mean(axis=(1, 2, 3)) / 255.0)
        label.append(lab[: len(frames) - 1])
        source.append(np.full(len(frames) - 1, name, dtype=object))
    motion = np.concatenate(motion)
    label = np.concatenate(label)
    source = np.concatenate(source)

    moving = label != 0
    still = ~moving
    print("共 %d 个 (obs_t, obs_t+1) 转移对，来自 %d 个来源" % (len(label), len(sources)))
    print("标签分布：不动 %.1f%%（= noop 类）| 其余 %d 个动作类共 %.1f%%"
          % (100 * still.mean(), len(np.unique(label)) - 1, 100 * moving.mean()))

    print("\n=== 画面变化量（mean|Δ|/255）按标签分组 ===")
    print("%-14s %7s %7s %7s %7s" % ("标签", "中位", "p10", "p90", "样本"))
    for tag, sel in (("不动 noop", still), ("有动作", moving)):
        m = motion[sel]
        print("%-14s %7.4f %7.4f %7.4f %7d"
              % (tag, np.median(m), np.percentile(m, 10), np.percentile(m, 90), sel.sum()))

    # 「动了就说明按了键」到底能区分到什么程度：AUC=1 完美，0.5 等于瞎猜。
    a = auc(motion, moving)
    print("\n用画面变化量区分「有动作 vs 不动」的 AUC = %.3f（1.0 完美，0.5 瞎猜）" % a)
    print("⇒ 两个分布重叠 %.0f%%（重叠部分无论阈值怎么切都会切错）" % (100 * (1 - abs(2 * a - 1))))

    # 取让平衡准确率最大的阈值 —— 不选任意阈值，选对模型最有利的那个。
    grid = np.quantile(motion, np.linspace(0.02, 0.98, 97))
    best = max((( 0.5 * ((motion[still] <= t).mean() + (motion[moving] > t).mean()), t)
                for t in grid), key=lambda p: p[0])
    t = best[1]
    print("\n=== 在最有利阈值 %.4f 下的 2×2（%s 是「画面没变」一侧）==="
          % (t, "上排"))
    print("%-22s %10s %10s" % ("", "标签=不动", "标签=有动作"))
    print("%-22s %10.1f%% %9.1f%%" % ("画面没变", 100 * (still & (motion <= t)).mean(),
                                      100 * (moving & (motion <= t)).mean()))
    print("%-22s %10.1f%% %9.1f%%" % ("画面变了", 100 * (still & (motion > t)).mean(),
                                      100 * (moving & (motion > t)).mean()))
    conflict = float((moving & (motion <= t)).mean() + (still & (motion > t)).mean())
    print("\n**不可辨识对 = %.1f%%**（画面没变却按了键 + 画面变了却没按键）" % (100 * conflict))
    print("这部分无论什么模型都只能靠猜；纯像素模型在这个数据上的准确率上界 ≈ %.1f%%"
          % (100 * (1 - conflict)))

    print("\n=== 「画面没变」里的标签构成（前 8 类）===")
    cats, counts = np.unique(label[motion <= t], return_counts=True)
    order = np.argsort(-counts)[:8]
    for i in order:
        print("  %-28s %6.1f%%  (%d)" % (names[int(cats[i])], 100 * counts[i] / (motion <= t).sum(),
                                          counts[i]))
    print("\n=== 「画面变了」里的标签构成（前 8 类）===")
    cats2, counts2 = np.unique(label[motion > t], return_counts=True)
    for i in np.argsort(-counts2)[:8]:
        print("  %-28s %6.1f%%  (%d)" % (names[int(cats2[i])],
                                          100 * counts2[i] / (motion > t).sum(), counts2[i]))

    print("\n=== 逐来源的不可辨识占比（分母 = 该来源自己的对数）===")
    for name in sources:
        sel = source == name
        # Both terms must be shares of the SAME denominator.  Adding two shares
        # of different-sized subsets is how an earlier version of this line
        # printed "104.1%".
        c = float(((label[sel] != 0) & (motion[sel] <= t)).mean()
                  + ((label[sel] == 0) & (motion[sel] > t)).mean())
        print("  %-16s %5d 对 | 不可辨识 %5.1f%% | 真 noop %5.1f%%"
              % (name, sel.sum(), 100 * c, 100 * (label[sel] == 0).mean()))

    # 不可辨识有两种成因，它们指向完全不同的修法：
    #   (a) 语义歧义（等剧情/平台/顶墙）—— 标签是对的，画面本来就不含信息；
    #   (b) 标签与画面错位一帧 —— 修法是重新对齐，不是换模型。
    # 把标签整体平移几个位置，看冲突率的最小值在不在 0：不在 0 就是 (b)。
    print("\n=== 标签整体平移后的冲突率（最小值不在 shift=0 就说明错位）===")
    print("%8s %10s" % ("shift", "不可辨识"))
    for shift in range(-3, 4):
        if shift > 0:                       # 标签滞后：拿 t 的画面配 t-shift 的按键
            mot_s, lab_s = motion[: len(label) - shift], label[shift:]
        elif shift < 0:
            mot_s, lab_s = motion[-shift:], label[: len(label) + shift]
        else:
            mot_s, lab_s = motion, label
        c = float(((lab_s != 0) & (mot_s <= t)).mean() + ((lab_s == 0) & (mot_s > t)).mean())
        print("%8d %9.1f%%" % (shift, 100 * c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
