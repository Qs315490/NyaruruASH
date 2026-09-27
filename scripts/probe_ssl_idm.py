"""Can the corpus, used WITHOUT labels, teach the IDM to transfer across scenes?

Everything measured so far says the IDM's action head memorises a scene and answers
its prior in a new one (ImpalaCNN 45.5%, frozen DINOv2 + head 45.1%, against a 50.4%
majority baseline). Its supervised data is ~5k labelled pairs from four rooms, which
is why.

The corpus is 68k frames across dozens of scenes and has no action labels - but it
does have TIME, and "what changed between these two frames" is exactly what a
cross-scene motion prior is made of. So pretrain the trunk with temporal
discrimination (is this pair adjacent, or far apart?), which needs no labels at all,
then fine-tune the action head on the labelled recordings and re-measure the same
cross-scene criterion.

Pass/fail is unchanged and must not be softened: held-out scene accuracy clearly
above the majority baseline.

    uv run python scripts/probe_ssl_idm.py
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
from ash.data.corpus_crop import CroppedFrames, load_crops  # noqa: E402
from ash.loop.bootstrap import _prep  # noqa: E402
from ash.models.idm import IdmConfig, IdmModel  # noqa: E402
from ash.train.demo_mapping import load_demo, map_legacy_masks  # noqa: E402
from ash.utils.device import resolve_device  # noqa: E402

SIZE = 128
CORPUS_STEMS = ["0r2lVc1uKa0", "BV19s4y1y7un"]
PER_VIDEO = 6000
STRIDE = 3          # the corpus is ~8 fps; stride 3 lands near the agent's 0.25 s
GAP = 40            # "far apart" for a negative pair
NAMES = ["house-001", "house-002", "house-003", "human-001"]
HOLDOUT = "human-001"


def load_video_frames(stem: str) -> np.ndarray:
    """A strided block of one corpus video, already cropped and at SIZE."""
    rect = load_crops("data/corpus-crops.json")[stem]
    arr = np.load(f"data/corpus/{stem}.npy", mmap_mode="r")
    take = np.arange(0, len(arr), STRIDE)[:PER_VIDEO]
    out = np.empty((len(take), SIZE, SIZE, 3), dtype=np.uint8)
    for i, frame in enumerate(CroppedFrames(arr, rect)[take]):
        out[i] = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return out


def temporal_pairs(frames: np.ndarray, offset: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Balanced adjacent / far-apart pairs; returns (a, b, is_adjacent)."""
    n = len(frames) - GAP - 1
    half = n // 2
    idx = rng.permutation(n)[: half * 2]
    pos, neg = idx[:half], idx[half:]
    a = np.concatenate([frames[pos], frames[neg]])
    b = np.concatenate([frames[pos + 1], frames[neg + GAP]])
    y = np.concatenate([np.ones(half, np.float32), np.zeros(half, np.float32)])
    order = rng.permutation(len(y))
    return a[order], b[order], y[order]


def batched(frames: np.ndarray, size: int = 64):
    for i in range(0, len(frames), size):
        yield torch.from_numpy(_prep(frames[i : i + size], SIZE))


def main() -> int:
    dev = resolve_device("cuda")
    rng = np.random.default_rng(0)
    space = ActionSpace.minimal()

    print("载入语料（无标签）...")
    corpus = np.concatenate([load_video_frames(s) for s in CORPUS_STEMS])
    print("  语料 %d 帧，来自 %d 个视频（跨多个场景，无动作标签）" % (len(corpus), len(CORPUS_STEMS)))

    model = IdmModel(IdmConfig(image_size=SIZE, num_actions=len(space))).to(dev)
    # --- 自监督：判断两帧是否相邻 -------------------------------------------
    probe = nn.Linear(model.config.embed_dim * 3, 1).to(dev)
    opt = torch.optim.AdamW(list(model.trunk.parameters()) + list(probe.parameters()), lr=3e-4)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(6):
        a, b, y = temporal_pairs(corpus, GAP, rng)
        total, seen = 0.0, 0
        for ta, tb, ty in zip(batched(a, 128), batched(b, 128),
                              torch.split(torch.from_numpy(y), 128)):
            ta, tb, ty = ta.to(dev), tb.to(dev), ty.to(dev)
            ea = model.trunk(ta.unsqueeze(1)).squeeze(1)
            eb = model.trunk(tb.unsqueeze(1)).squeeze(1)
            logit = probe(torch.cat([ea, eb, eb - ea], dim=1)).squeeze(1)
            loss = bce(logit, ty)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(ty); seen += len(ty)
        print("  ssl epoch %d: loss %.4f (相邻/相隔二分类；0.693 = 瞎猜)" % (epoch, total / seen))

    del probe

    # --- 监督微调：只用带标签的录制 -----------------------------------------
    feats = {}
    for name in NAMES:
        d = load_demo(f"data/recordings/{name}.npz")
        idx, _ = map_legacy_masks(np.asarray(d["control_masks"]), space)
        obs = np.asarray(d["observations"])
        feats[name] = (obs, idx[1:].astype(np.int64))
        print("  %-11s %5d 帧" % (name, len(obs)))

    def fine_tune(model, epochs=6):
        pairs_a, pairs_b, labels = [], [], []
        for name in NAMES:
            if name == HOLDOUT:
                continue
            obs, lab = feats[name]
            pairs_a.append(obs[:-1]); pairs_b.append(obs[1:]); labels.append(lab)
        a = np.concatenate(pairs_a); b = np.concatenate(pairs_b)
        y = torch.from_numpy(np.concatenate(labels))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
        ce = nn.CrossEntropyLoss()
        model.train()
        order = np.arange(len(a))
        for epoch in range(epochs):
            rng.shuffle(order)
            for i in range(0, len(order), 8):
                take = order[i : i + 8]
                ta = torch.from_numpy(_prep(a[take], SIZE)).to(dev)
                tb = torch.from_numpy(_prep(b[take], SIZE)).to(dev)
                loss = ce(model(ta, tb), y[take].to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
            print("  ft epoch %d: loss %.4f" % (epoch, float(loss)))
        return model

    print("监督微调（仅 3 个屋子录制）...")
    model = fine_tune(model)

    model.eval()
    obs, lab = feats[HOLDOUT]
    preds = []
    with torch.inference_mode():
        for i in range(0, len(obs) - 1, 64):
            j = min(i + 64, len(obs) - 1)
            ta = torch.from_numpy(_prep(obs[i:j], SIZE)).to(dev)
            tb = torch.from_numpy(_prep(obs[i + 1 : j + 1], SIZE)).to(dev)
            preds.append(model(ta, tb).argmax(-1).cpu().numpy())
    pred = np.concatenate(preds)
    t = lab[: len(pred)]
    base = max((t == 0).mean(), (t != 0).mean())
    acc = (pred == t).mean()
    print("\n留出跨场景 %s: 准确率 %.1f%% | 多数类基线 %.1f%% | 预测noop %.1f%% | 真noop %.1f%%" % (
        HOLDOUT, 100 * acc, 100 * base, 100 * (pred == 0).mean(), 100 * (t == 0).mean()))
    print("=> %s" % ("通过：自监督预训练让动作头跨场景泛化" if acc > base + 0.05
                     else "未通过：自监督预训练也没能带来跨场景泛化"))
    print("（对照：不预训练的同结构 IDM 为 45.5%% vs 基线 50.4%%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
