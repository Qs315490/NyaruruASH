"""Does the training data's COMPOSITION explain the failure, or its size?

The label-identifiability probe measured what the pairs actually are:

    label=noop   label=move
    still         42.7%      10.3%     <- pressing into a wall
    moving        24.8%      22.2%     <- the world moved, not the player

Only the bottom-right cell is both clean and informative, but 42.7% of the set
teaches "do not move", and a policy that answers noop everywhere therefore looks
respectable on accuracy.  The question this script answers: if the model is fed
only pairs whose label the pixels can actually determine, does per-class recall
leave chance?

Four arms, all evaluated on the held-out room's *informative* pairs only (a
test set made of movement labels, where a noop-answering model scores ~0):

  A  all training pairs                    -- the current behaviour
  B  informative pairs only                -- reads "which key" with no noop at all
  C  a random subset with B's pair count   -- controls for the smaller data size
  D  informative + equally many static noop pairs -- the reweighted version

B >> C would mean composition is the lever; B ~= C would mean size is.

    uv run python scripts/probe_informative_subset.py
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
LR = 3e-5
GRAD_CLIP = 1.0

GROUPS: dict[str, list[str]] = {
    "A:ep00-01": ["human-001:ep00", "human-001:ep01"],
    "B:ep02-03": ["human-001:ep02", "human-001:ep03"],
    "C:ep05-06-10": ["human-001:ep05", "human-001:ep06", "human-001:ep10"],
    "D:ep07-08-09": ["human-001:ep07", "human-001:ep08", "human-001:ep09"],
    "E:ep11-12": ["human-001:ep11", "human-001:ep12"],
    "F:ep04": ["human-001:ep04"],
    "G:house": ["house-001", "house-002", "house-003"],
}


def load_sources() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    space = ActionSpace.minimal()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    demo = Path("data/idm-human")
    obs = np.load(demo / "observations.npy", mmap_mode="r")
    eps = np.load(demo / "episode_ids.npy", mmap_mode="r")
    masks = np.load(demo / "control_masks.npy", mmap_mode="r")
    for e in np.unique(eps):
        rows = np.flatnonzero(eps == e)
        index, _ = map_legacy_masks(np.asarray(masks[rows]), space)
        out[f"human-001:ep{int(e):02d}"] = (np.asarray(obs[rows]), index)
    for name in ("house-001", "house-002", "house-003"):
        z = np.load(f"data/recordings/{name}.npz")
        o = z["observations"]
        small = np.stack([cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for f in o])
        index, _ = map_legacy_masks(np.asarray(z["control_masks"]), space)
        out[name] = (small, index)
    return out


def pairs_of(sources, names: list[str]):
    a, b, y = [], [], []
    for n in names:
        frames, lab = sources[n]
        if len(frames) < 2:
            continue
        a.append(frames[:-1])
        b.append(frames[1:])
        y.append(lab[: len(frames) - 1])
    return np.concatenate(a), np.concatenate(b), np.concatenate(y)


def motion_of(a, b) -> np.ndarray:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return d.mean(axis=(1, 2, 3)) / 255.0


def pick_threshold(motion: np.ndarray, moving: np.ndarray) -> float:
    """The cut that separates noop from movement best, chosen on TRAINING data.

    Fixing the cut by hand would decide the result; taking the best one is the
    most generous choice available, for any threshold, to the model.
    """
    grid = np.quantile(motion, np.linspace(0.02, 0.98, 97))
    return float(max(grid, key=lambda t: 0.5 * (((~moving) & (motion <= t)).mean()
                                                + (moving & (motion > t)).mean())))


def train_eval(dev, tr, te, label: str) -> dict:
    a_tr, b_tr, y_tr = tr
    a_te, b_te, y_te = te
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
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += float(loss) * len(chunk); seen += len(chunk)
        print("    [%s] epoch %d train %.4f" % (label, epoch, total / max(1, seen)), flush=True)

    preds = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(a_te), 64):
            x = torch.from_numpy(_prep(a_te[i : i + 64], SIZE)).to(dev)
            z = torch.from_numpy(_prep(b_te[i : i + 64], SIZE)).to(dev)
            preds.append(model(x, z).argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    classes = np.unique(y_te)
    recalls = [float(((pred == c) & (y_te == c)).sum() / max(1, (y_te == c).sum())) for c in classes]
    acc = float((pred == y_te).mean())
    chance = 1.0 / len(classes)
    print("    [%s] 有信息留出 %d 对 | 准确率 %.1f%% | **macro-recall %.3f**（随机 %.3f）"
          " | 预测noop %.1f%% | 训练 %d 对"
          % (label, len(y_te), 100 * acc, float(np.mean(recalls)), chance,
             100 * (pred == 0).mean(), len(y_tr)), flush=True)
    return {"acc": acc, "recall": float(np.mean(recalls)), "chance": chance,
            "pred_noop": float((pred == 0).mean()), "n_test": len(y_te),
            "n_train": len(y_tr)}


def main() -> int:
    dev = resolve_device("cuda")
    sources = load_sources()
    results: dict[str, list[dict]] = {}

    for group in GROUPS:
        train_names = [n for g, ns in GROUPS.items() if g != group for n in ns]
        a_tr, b_tr, y_tr = pairs_of(sources, train_names)
        a_te, b_te, y_te = pairs_of(sources, GROUPS[group])
        m_tr, m_te = motion_of(a_tr, b_tr), motion_of(a_te, b_te)
        thr = pick_threshold(m_tr, y_tr != 0)
        info_tr = (m_tr > thr) & (y_tr != 0)
        info_te = (m_te > thr) & (y_te != 0)
        static_noop_tr = (m_tr <= thr) & (y_tr == 0)
        print("\n留出 %s | 阈值 %.4f（在训练集上选） | 训练 %d 对其中可辨识 %d 对（%.1f%%）"
              " | 留出有信息 %d 对"
              % (group, thr, len(y_tr), int(info_tr.sum()), 100 * info_tr.mean(),
                 int(info_te.sum())), flush=True)

        rng = np.random.default_rng(0)
        n_info = int(info_tr.sum())
        rand_subset = np.zeros(len(y_tr), dtype=bool)
        rand_subset[rng.permutation(len(y_tr))[:n_info]] = True
        n_static = int(static_noop_tr.sum())
        take_static = np.zeros(len(y_tr), dtype=bool)
        if n_static:
            take_static[rng.permutation(np.flatnonzero(static_noop_tr))[:n_info]] = True

        arms = {
            "A 全量": np.ones(len(y_tr), dtype=bool),
            "B 只喂有信息": info_tr,
            "C 随机等量": rand_subset,
            "D 有信息+等量静止noop": info_tr | take_static,
        }
        for name, sel in arms.items():
            if sel.sum() < 2:
                print("    [%s] 样本不足，跳过" % name)
                continue
            results.setdefault(name, []).append(train_eval(
                dev,
                (a_tr[sel], b_tr[sel], y_tr[sel]),
                (a_te[info_te], b_te[info_te], y_te[info_te]),
                "%s %s" % (group, name)))

    print("\n=== 汇总（留出有信息的运动帧；随机 ≈ 1/类数）===")
    print("%-24s %8s %10s %12s %10s" % ("臂", "训练对数", "准确率", "macro-recall", "预测noop"))
    for name, rs in results.items():
        print("%-24s %8.0f %9.1f%% %12.3f %9.1f%%"
              % (name, np.mean([r["n_train"] for r in rs]),
                 100 * np.mean([r["acc"] for r in rs]),
                 np.mean([r["recall"] for r in rs]),
                 100 * np.mean([r["pred_noop"] for r in rs])))
    print("（留出集全是运动帧，所以「预测noop」只要不为 0，那些样本必然全错；"
          "随机水平的 macro-recall ≈ 1/类数，逐折日志里有）")
    print("\n判读：B 明显高于 C ⇒ 组成（配重）是杠杆；B ≈ C ⇒ 只是数据量在起作用；"
          "\n      B 与 A 都停在随机 ⇒ 只喂干净数据也不够，像素本身推不出按键。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
