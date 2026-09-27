"""Does feeding the trunk the difference image let it express motion?

Diagnosis first (measured, all four checks in one run):

  (a) labels vary                       ok
  (b) trunk features vary with input    ok
  (c) frozen features + linear head, held-out split   49.0%  = chance
  (d) gradient reaches the trunk        ok
  (e) joint training, shuffled batches  ~50-58% = chance

and the task itself is separable: adjacent frames differ by a median |delta| of 23.7
against 56.4 for frames 40 apart - a 2.4x gap that a bare threshold on the pixel
difference turns into 79.2% accuracy.  So the features are capable of nothing here
because the structure discards the signal: `[ea, eb, eb - ea]` is built from
GLOBAL-AVERAGE-POOLED vectors, and motion is spatial and local - averaging each frame
first, then subtracting, flattens it away.

The IDM uses exactly that structure for its action head, which is a second reason it
only memorises scenes.  This runs the same temporal task with the difference image
itself as input, held-out split.

    uv run python scripts/probe_motion_head.py
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
from ash.utils.device import resolve_device  # noqa: E402

SIZE = 128
FAR = 40          # "far apart" for a negative pair, in strided frames
PAIRS = 1500      # per class
HOLDOUT = 800
BATCH = 128


def main() -> int:
    dev = resolve_device("cuda")
    rng = np.random.default_rng(0)
    rect = load_crops("data/corpus-crops.json")["0r2lVc1uKa0"]
    arr = np.load("data/corpus/0r2lVc1uKa0.npy", mmap_mode="r")
    frames = np.stack([cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                       for f in CroppedFrames(arr, rect)[::3][:3000]])
    limit = len(frames) - FAR - 1
    per = min(PAIRS, limit // 2)
    idx = rng.permutation(limit)
    pos, neg = idx[:per], idx[per: 2 * per]
    a = np.concatenate([frames[pos], frames[neg]])
    b = np.concatenate([frames[pos + 1], frames[neg + FAR]])
    y = np.concatenate([np.ones(per, np.float32), np.zeros(per, np.float32)])
    order = rng.permutation(len(y))
    a, b, y = a[order], b[order], y[order]
    print("%d 对（正 %d / 负 %d，%d 帧素材）" % (len(y), per, per, len(frames)))

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    plain = np.concatenate([a, b], axis=0)          # 对照：不做差，只把两帧堆一起

    model = IdmModel(IdmConfig(image_size=SIZE, num_actions=len(ActionSpace.minimal()))).to(dev)
    head = nn.Linear(model.config.embed_dim, 1).to(dev)
    opt = torch.optim.AdamW(list(model.trunk.parameters()) + list(head.parameters()), lr=1e-3)
    train_n = len(y) - HOLDOUT

    def run(source: np.ndarray, lo: int, hi: int, train: bool) -> float:
        accs = []
        for i in range(lo, hi, BATCH):
            j = min(i + BATCH, hi)
            x = torch.from_numpy(_prep(source[i:j], SIZE)).to(dev)
            t = torch.from_numpy(y[i:j]).to(dev)
            logit = head(model.trunk(x.unsqueeze(1)).squeeze(1)).squeeze(1)
            accs.append(float(((logit > 0).float() == t).float().mean()))
            if train:
                loss = nn.functional.binary_cross_entropy_with_logits(logit, t)
                opt.zero_grad(); loss.backward(); opt.step()
        return 100.0 * float(np.mean(accs))

    print("输入 = 差异图 |a-b|:")
    best = 0.0
    for epoch in range(8):
        tr = run(diff, 0, train_n, True)
        with torch.no_grad():
            te = run(diff, train_n, len(y), False)
        best = max(best, te)
        print("  epoch %d: 训练 %.1f%% | 留出 %.1f%%" % (epoch, tr, te))
    print("=> %s" % ("通过：差异图让 trunk 学到了运动" if best > 65.0
                     else "未通过：换成差异图也没学到"))
    print("（对照：GAP 特征做差的留出准确率是 49.0%，多数类基线 50%；"
          "像素差阈值的上限是 79.2%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
