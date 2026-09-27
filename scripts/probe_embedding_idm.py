"""Would a frozen DINOv2 encoder + a small head generalise across scenes?

The shipped IDM trains an ImpalaCNN from scratch on a few thousand frame pairs.
Trained on the three house recordings it scores 45.5% on a held-out recording whose
majority class is 50.4% - i.e. it memorises the scene and answers its prior
anywhere else, which is the same prior the corpus pseudo-labels consist of
(~50% noop).  DINOv2 features are the one thing in this project that already
transfers across domains (K and retrieval run on them, and after the crop a live
frame sits at 0.94 cosine to a video frame), so this probes the cheap half of the
question before refactoring the model.

    uv run python scripts/probe_embedding_idm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.actions.space import ActionSpace  # noqa: E402
from ash.memory.embeddings import FrameEmbedder  # noqa: E402
from ash.train.demo_mapping import load_demo, map_legacy_masks  # noqa: E402

NAMES = ["house-001", "house-002", "house-003", "human-001"]
HOLDOUT = "human-001"


def main() -> int:
    space = ActionSpace.minimal()
    emb = FrameEmbedder(device="cuda", image_size=256)
    feats: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for name in NAMES:
        demo = load_demo(f"data/recordings/{name}.npz")
        obs = np.asarray(demo["observations"])
        index, _ = map_legacy_masks(np.asarray(demo["control_masks"]), space)
        feats[name] = emb.embed(obs)
        labels[name] = index[1:].astype(np.int64)
        print("%-11s %5d 帧" % (name, len(obs)))

    def pairs(name: str) -> tuple[np.ndarray, np.ndarray]:
        e = feats[name]
        x = np.concatenate([e[:-1], e[1:], e[1:] - e[:-1]], axis=1)
        return x, labels[name]

    xtr = np.concatenate([pairs(n)[0] for n in NAMES if n != HOLDOUT])
    ytr = np.concatenate([pairs(n)[1] for n in NAMES if n != HOLDOUT])
    xte, yte = pairs(HOLDOUT)
    print("训练 %d 对 | 留出 %s %d 对 | 真 noop %.1f%%" % (
        len(ytr), HOLDOUT, len(yte), 100.0 * (yte == 0).mean()))

    torch.manual_seed(0)
    head = torch.nn.Sequential(
        torch.nn.Linear(x_tr := xtr.shape[1], 256), torch.nn.ReLU(),
        torch.nn.Linear(256, len(space)),
    )
    assert x_tr == xte.shape[1]
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    xb = torch.from_numpy(xtr.astype(np.float32))
    yb = torch.from_numpy(ytr)
    ce = torch.nn.CrossEntropyLoss()
    for epoch in range(40):
        perm = torch.randperm(len(xb))
        total = 0.0
        for i in range(0, len(perm), 256):
            take = perm[i:i + 256]
            loss = ce(head(xb[take]), yb[take])
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(take)
        if epoch % 10 == 9:
            print("  epoch %2d loss %.4f" % (epoch, total / len(xb)))

    with torch.inference_mode():
        pred = head(torch.from_numpy(xte.astype(np.float32))).argmax(-1).numpy()
    baseline = max((yte == 0).mean(), (yte != 0).mean())
    print("留出跨场景准确率 %.1f%% | 多数类基线 %.1f%% | 预测noop %.1f%% | 真noop %.1f%%" % (
        100.0 * (pred == yte).mean(), 100.0 * baseline,
        100.0 * (pred == 0).mean(), 100.0 * (yte == 0).mean()))
    print("=> %s" % ("通过（显著高于基线）" if (pred == yte).mean() > baseline + 0.05
                     else "未通过：嵌入头也没能跨场景泛化"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
