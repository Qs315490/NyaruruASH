"""Is the IDM blocked by *how many rooms* it has seen, or by the task itself?

The probe that motivated this compared "train 3 house recordings, hold out
human-001" and got 45.5% against a 50.4% majority baseline.  It is easy to read
that as "the labelled set is small".  It is not: the three house recordings are
three sessions of the SAME room (frame-level cosine 0.995-0.998), so that probe
trained on one room and asked for a second one.  The labelled corpus actually
holds several distinct rooms -- human-001's 13 episodes fall into groups whose
frames match at 0.95-0.997, i.e. the same place, while different groups sit at
0.60-0.87.

So the question this script answers is the one that decides where to spend the
next effort:

    does cross-room accuracy rise as the training set gains MORE ROOMS?

  * it rises        -> the model is data-limited; more coverage is the lever,
                       and no architecture change will move the number.
  * it stays flat   -> at this data scale the pixels do not determine the key
                       across rooms at all.  Then neither a new architecture nor
                       a new pretraining objective helps, and the problem has to
                       be reformulated (self-play labels, or a distribution gate).

Two controls keep the answer honest:

* the same-room ceiling (temporal 70/30 split *inside* one room) -- a random
  split leaks near-duplicate neighbours and is how an earlier probe reached a
  flattering 91.6%;
* the sweep is run twice, once with the training pairs growing and once with
  them capped to a fixed count, because "more rooms" and "more pairs" are
  otherwise confounded.

    uv run python scripts/probe_room_generalization.py
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
EPOCHS = 3
BATCH = 8
#: Matches BootstrapConfig.lr, which is what the in-loop update uses.  That rate
#: is chosen for FINE-TUNING a checkpoint that `pretrain-idm` already fitted at
#: 3e-4; training from scratch at 3e-5 is a different experiment, so `--lr`
#: exists to tell the two apart instead of quietly reporting the wrong one.
LR = 3e-5
GRAD_CLIP = 1.0
SKIP_SWEEP = False

# Groups come from the measured DINOv2 similarity matrix, not from filenames
# (see the docstring): single-linkage on "max frame cosine >= 0.95".
GROUPS: dict[str, list[str]] = {
    "A:ep00-01": ["human-001:ep00", "human-001:ep01"],
    "B:ep02-03": ["human-001:ep02", "human-001:ep03"],
    "C:ep05-06-10": ["human-001:ep05", "human-001:ep06", "human-001:ep10"],
    "D:ep07-08-09": ["human-001:ep07", "human-001:ep08", "human-001:ep09"],
    "E:ep11-12": ["human-001:ep11", "human-001:ep12"],
    "F:ep04": ["human-001:ep04"],
    "G:house": ["house-001", "house-002", "house-003"],
}
HELD_OUT_FOR_SWEEP = "E:ep11-12"
CAP = 4000


def load_sources() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{source name: (frames @128, action index per transition)}."""
    space = ActionSpace.minimal()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    demo = Path("data/idm-human")
    obs = np.load(demo / "observations.npy", mmap_mode="r")
    eps = np.load(demo / "episode_ids.npy", mmap_mode="r")
    masks = np.load(demo / "control_masks.npy", mmap_mode="r")
    for e in np.unique(eps):
        rows = np.flatnonzero(eps == e)
        index, _ = map_legacy_masks(np.asarray(masks[rows]), space)
        out[f"human-001:ep{int(e):02d}"] = (np.asarray(obs[rows]), index[1:].astype(np.int64))
    for name in ("house-001", "house-002", "house-003"):
        z = np.load(f"data/recordings/{name}.npz")
        o = z["observations"]
        small = np.stack([cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for f in o])
        index, _ = map_legacy_masks(np.asarray(z["control_masks"]), space)
        out[name] = (small, index[1:].astype(np.int64))
    return out


def pairs_of(sources, names: list[str]):
    """Concatenate *whole* sources, so no phantom transition crosses a seam."""
    a, b, y = [], [], []
    for n in names:
        frames, lab = sources[n]
        if len(frames) < 2:
            continue
        a.append(frames[:-1])
        b.append(frames[1:])
        y.append(lab)
    return np.concatenate(a), np.concatenate(b), np.concatenate(y)


def train_eval(dev, a_tr, b_tr, y_tr, a_te, b_te, y_te, label: str) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)
    model = IdmModel(IdmConfig(image_size=SIZE, num_actions=len(ActionSpace.minimal()))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    order = np.arange(len(a_tr))
    rng = np.random.default_rng(0)
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        total, seen = 0.0, 0
        for i in range(0, len(order), BATCH):
            chunk = order[i : i + BATCH]
            x = torch.from_numpy(_prep(a_tr[chunk], SIZE)).to(dev)
            z = torch.from_numpy(_prep(b_tr[chunk], SIZE)).to(dev)
            t = torch.from_numpy(y_tr[chunk]).to(dev)
            loss = ce(model(x, z), t)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += float(loss) * len(chunk)
            seen += len(chunk)
        print("    [%s] epoch %d train %.4f" % (label, epoch, total / max(1, seen)), flush=True)

    def predict(a, b):
        preds = []
        model.eval()
        with torch.inference_mode():
            for i in range(0, len(a), 64):
                x = torch.from_numpy(_prep(a[i : i + 64], SIZE)).to(dev)
                z = torch.from_numpy(_prep(b[i : i + 64], SIZE)).to(dev)
                preds.append(model(x, z).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    def metrics(y, preds) -> dict:
        """Overall accuracy hides the whole story here.

        The demonstrations are 50-100% noop, so a model that only ever answers
        noop already scores the majority baseline.  The number that says whether
        the action mapping was learned at all is accuracy *on the moving frames*,
        so it is reported separately, along with how often the model bothers to
        predict a movement.
        """
        majority = float(np.bincount(y, minlength=len(ActionSpace.minimal())).max() / len(y))
        moving = y != 0
        recalls = [float(((preds == c) & (y == c)).sum() / max(1, (y == c).sum()))
                   for c in np.unique(y)]
        return {
            "acc": float((preds == y).mean()),
            "baseline": majority,
            "acc_moving": float((preds[moving] == y[moving]).mean()) if moving.any() else float("nan"),
            "pred_moving": float((preds != 0).mean()),
            "true_moving": float(moving.mean()),
            "macro_recall": float(np.mean(recalls)) if recalls else float("nan"),
            "n_classes": int(len(np.unique(y))),
        }

    tr = metrics(y_tr[:2000], predict(a_tr[:2000], b_tr[:2000]))
    te = metrics(y_te, predict(a_te, b_te))
    print("    [%s] 训练 %.1f%% | 留出 %.1f%%（基线 %.1f%%）| **运动帧上 %.1f%%** | "
          "预测运动 %.1f%%（真 %.1f%%）| macro-recall %.3f | 类 %d"
          % (label, 100 * tr["acc"], 100 * te["acc"], 100 * te["baseline"],
             100 * te["acc_moving"], 100 * te["pred_moving"],
             100 * te["true_moving"], te["macro_recall"], te["n_classes"]), flush=True)
    return {"train_acc": tr["acc"], "test_acc": te["acc"], "baseline": te["baseline"],
            "acc_moving": te["acc_moving"], "pred_moving": te["pred_moving"],
            "true_moving": te["true_moving"], "macro_recall": te["macro_recall"],
            "n_train": len(y_tr), "n_test": len(y_te),
            "classes_in_test": te["n_classes"]}


def main() -> int:
    global LR, SKIP_SWEEP
    argv = [a for a in sys.argv[1:]]
    if "--lr" in argv:
        LR = float(argv[argv.index("--lr") + 1])
    SKIP_SWEEP = "--no-sweep" in argv
    print("超参: lr=%g epochs=%d batch=%d size=%d（生产 lr 见 BootstrapConfig）"
          % (LR, EPOCHS, BATCH, SIZE), flush=True)
    dev = resolve_device("cuda")
    sources = load_sources()
    for name, (frames, lab) in sources.items():
        print("  %-18s %5d 帧 | noop %.1f%%" % (name, len(frames), 100 * (lab == 0).mean()))

    results: dict[str, dict] = {}

    print("\n=== 1. 同房间天花板（组内时间序 70/30，随机划分会泄漏近邻帧）===")
    for group, names in GROUPS.items():
        before, after, labels = pairs_of(sources, names)
        cut = int(0.7 * len(labels))
        results["同房间 " + group] = train_eval(
            dev,
            before[:cut], after[:cut], labels[:cut],
            before[cut:], after[cut:], labels[cut:],
            "same:" + group,
        )

    print("\n=== 2. 跨房间：留出一个组，训练集 = 其余全部组 ===")
    for group in GROUPS:
        train_names = [n for g, ns in GROUPS.items() if g != group for n in ns]
        a_tr, b_tr, y_tr = pairs_of(sources, train_names)
        a_te, b_te, y_te = pairs_of(sources, GROUPS[group])
        print("  留出 %s | 训练 %d 对（%d 个组）| 留出 %d 对"
              % (group, len(y_tr), len(GROUPS) - 1, len(y_te)), flush=True)
        results["跨房间 " + group] = train_eval(
            dev, a_tr, b_tr, y_tr, a_te, b_te, y_te, "cross:" + group)

    if SKIP_SWEEP:
        print("\n（--no-sweep：跳过房间数扫描）")
        return _summary(results)

    print("\n=== 3. 房间数扫描（留出 %s；同一组对，只改训练组数）===" % HELD_OUT_FOR_SWEEP)
    order = [g for g in GROUPS if g != HELD_OUT_FOR_SWEEP]
    a_te, b_te, y_te = pairs_of(sources, GROUPS[HELD_OUT_FOR_SWEEP])
    for k in (1, 2, 3, len(order)):
        names = [n for g in order[:k] for n in GROUPS[g]]
        a_tr, b_tr, y_tr = pairs_of(sources, names)
        for cap_label, cap in (("不限", 0), ("固定 %d 对" % CAP, CAP)):
            if cap and len(y_tr) > cap:
                pick = np.random.default_rng(0).permutation(len(y_tr))[:cap]
                ac, bc, yc = a_tr[pick], b_tr[pick], y_tr[pick]
            else:
                ac, bc, yc = a_tr, b_tr, y_tr
            print("  k=%d 训练组 %s | %s | 训练 %d 对"
                  % (k, order[:k], cap_label, len(yc)), flush=True)
            results["扫描 k=%d %s" % (k, cap_label)] = train_eval(
                dev, ac, bc, yc, a_te, b_te, y_te,
                "k%d:%s" % (k, "cap" if cap else "all"))

    return _summary(results)


def _summary(results: dict) -> int:
    print("\n=== 汇总 ===")
    print("%-26s %7s %7s %7s %10s %9s %8s" % (
        "条件", "训练", "留出", "基线", "运动帧上", "预测运动", "recall"))
    for name, r in results.items():
        print("%-26s %6.1f%% %6.1f%% %6.1f%% %9.1f%% %8.1f%% %7.3f"
              % (name, 100 * r["train_acc"], 100 * r["test_acc"], 100 * r["baseline"],
                 100 * r["acc_moving"], 100 * r["pred_moving"], r["macro_recall"]))
    print("\n读法：「运动帧上」是**只看真标签非 noop 的留出帧**的准确率 —— "
          "只答 noop 的模型在这里是 0%，\n所以它才是「学到映射了吗」的判据；"
          "总体准确率被 50~100% 的 noop 占比撑起来了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
