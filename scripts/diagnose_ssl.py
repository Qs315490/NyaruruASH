"""Why did the self-supervised probe sit at loss = ln 2 and then go NaN?

Loss exactly ln 2 means the discriminator's logits were 0 for every sample, which
has only a few causes and they need different fixes:

  a. the labels never vary (a constant target's best prediction is the mean -> ln 2);
  b. the trunk's features do not vary with the input, so there is nothing to learn;
  c. the features vary but carry no information about adjacency;
  d. the gradients do not reach the trunk.

This prints the evidence for each, on a small batch, before any long run.

    uv run python scripts/diagnose_ssl.py
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
STEMS = ["0r2lVc1uKa0", "BV19s4y1y7un"]
PER_VIDEO = 1500
STRIDE = 3
GAP = 40


def load(stem: str) -> np.ndarray:
    rect = load_crops("data/corpus-crops.json")[stem]
    arr = np.load(f"data/corpus/{stem}.npy", mmap_mode="r")
    take = np.arange(0, len(arr), STRIDE)[:PER_VIDEO]
    return np.stack([cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                     for f in CroppedFrames(arr, rect)[take]])


def main() -> int:
    dev = resolve_device("cuda")
    rng = np.random.default_rng(0)
    space = ActionSpace.minimal()
    videos = {s: load(s) for s in STEMS}
    frames = np.concatenate(list(videos.values()))
    print("帧: %s" % {s: v.shape for s, v in videos.items()})

    n = len(frames) - GAP - 1
    idx = rng.permutation(n)[: 2 * (n // 2)]
    half = len(idx) // 2
    pos, neg = idx[:half], idx[half:]
    a = np.concatenate([frames[pos], frames[neg]])
    b = np.concatenate([frames[pos + 1], frames[neg + GAP]])
    y = np.concatenate([np.ones(half, np.float32), np.zeros(half, np.float32)])
    print("(a) 标签: 均值 %.3f 唯一值 %s   ← 必须两者都有" % (y.mean(), np.unique(y)))

    model = IdmModel(IdmConfig(image_size=SIZE, num_actions=len(space))).to(dev)
    # 输入张量不能生在 inference_mode 里：那样它们带着 inference 标记，
    # 后面任何 autograd 都会抛 "Inference tensors cannot be saved for backward"。
    ta = torch.from_numpy(_prep(a[:256], SIZE)).to(dev)
    tb = torch.from_numpy(_prep(b[:256], SIZE)).to(dev)
    model.eval()
    with torch.no_grad():
        ea = model.trunk(ta.unsqueeze(1)).squeeze(1)
        eb = model.trunk(tb.unsqueeze(1)).squeeze(1)
    print("(b) trunk 输出: 形状 %s | 逐样本范数 中位 %.4f | 样本间输出标准差 %.5f"
          % (tuple(ea.shape), float(ea.norm(dim=1).median()), float(ea.std(dim=0).mean())))
    feat = torch.cat([ea, eb, eb - ea], dim=1)
    print("    特征在样本间是否有变化:", "有" if float(feat.std(dim=0).mean()) > 1e-6 else "没有 ← 这是病根")

    # (c) 冻结 trunk，只训一个线性判别器 —— 必须**带留出**，否则就是背题:
    # 之前在同一批 256 个样本上训又测，768 维特征可以背下任何标签，于是「100%」
    # 是记忆而不是信息。
    n_tr = 2000
    order0 = rng.permutation(len(y))
    tr_idx, te_idx = order0[:n_tr], order0[n_tr:n_tr + 800]
    # 分批 forward：一次推 2000 帧会让 ImpalaCNN 的激活要 8 GB，直接爆卡。
    def feats(frames: np.ndarray) -> torch.Tensor:
        out = []
        with torch.no_grad():
            for i in range(0, len(frames), 64):
                x = torch.from_numpy(_prep(frames[i:i + 64], SIZE)).to(dev)
                out.append(model.trunk(x.unsqueeze(1)).squeeze(1).cpu())
        return torch.cat(out).to(dev)

    ea_tr, eb_tr = feats(a[tr_idx]), feats(b[tr_idx])
    ea_te, eb_te = feats(a[te_idx]), feats(b[te_idx])
    x_tr = torch.cat([ea_tr, eb_tr, eb_tr - ea_tr], dim=1)
    x_te = torch.cat([ea_te, eb_te, eb_te - ea_te], dim=1)
    y_tr = torch.from_numpy(y[tr_idx]).to(dev)
    y_te = torch.from_numpy(y[te_idx]).to(dev)
    torch.manual_seed(0)
    lin = nn.Linear(x_tr.shape[1], 1).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(1000):
        lo = nn.functional.binary_cross_entropy_with_logits(lin(x_tr).squeeze(1), y_tr)
        opt.zero_grad(); lo.backward(); opt.step()
    with torch.no_grad():
        acc_tr = float(((lin(x_tr).squeeze(1) > 0).float() == y_tr).float().mean())
        acc_te = float(((lin(x_te).squeeze(1) > 0).float() == y_te).float().mean())
    print("(c) 冻结特征 + 线性判别: 训练 %d 样本 %.1f%% | **留出 %d 样本 %.1f%%** (50%% = 无信息)"
          % (n_tr, 100 * acc_tr, len(te_idx), 100 * acc_te))

    # (d) 梯度有没有到 trunk —— 特征必须在图内重算：
    # 之前的写法在 inference_mode 里算了 feat 再 backward，那是无梯度张量，
    # 于是「梯度为 0」测的是我的脚本，而不是模型。
    with torch.enable_grad():
        ea = model.trunk(ta.unsqueeze(1)).squeeze(1)
        eb = model.trunk(tb.unsqueeze(1)).squeeze(1)
        feat_g = torch.cat([ea, eb, eb - ea], dim=1)
    probe = nn.Linear(feat_g.shape[1], 1).to(dev)
    opt = torch.optim.AdamW(list(model.trunk.parameters()) + list(probe.parameters()), lr=1e-4)
    ty = torch.from_numpy(y[:256]).to(dev)
    loss = nn.functional.binary_cross_entropy_with_logits(probe(feat_g).squeeze(1), ty)
    opt.zero_grad(); loss.backward()
    grads = [p.grad for p in model.trunk.parameters() if p.grad is not None]
    gn = float(torch.sqrt(sum(g.pow(2).sum() for g in grads)))
    print("(d) trunk 梯度范数 %.3e | loss %.4f  ← 梯度为 0 则 (d) 是病根" % (gn, float(loss)))

    # (e0) 关键对照：batch 内必须混合正负样本。a/b/y 是按「正样本在前」拼的，
    # 不打散的话每个 batch 的标签全同，模型只能拟合一个常数，准确率在 0%/100% 之间
    # 跳、loss 恒在 ln 2 —— 那是测试的毛病，不是模型的。
    order = rng.permutation(len(y))
    a, b, y = a[order], b[order], y[order]

    # (e) 训练到底动没动：逐步打印，看学习率/步数够不够
    probe2 = nn.Linear(feat_g.shape[1], 1).to(dev)
    opt = torch.optim.AdamW(list(model.trunk.parameters()) + list(probe2.parameters()), lr=1e-3)
    # 分批准备：把 12k 帧一次性转成 float32 张量是 4.7 GB，直接把卡撑爆。
    print("(e) 打散后训练 400 步（lr 1e-3）:")
    for step in range(400):
        i = (step * 128) % (len(y) - 128)
        ta_i = torch.from_numpy(_prep(a[i:i+128], SIZE)).to(dev)
        tb_i = torch.from_numpy(_prep(b[i:i+128], SIZE)).to(dev)
        ty_i = torch.from_numpy(y[i:i+128]).to(dev)
        ea = model.trunk(ta_i.unsqueeze(1)).squeeze(1)
        eb = model.trunk(tb_i.unsqueeze(1)).squeeze(1)
        lo = nn.functional.binary_cross_entropy_with_logits(
            probe2(torch.cat([ea, eb, eb - ea], dim=1)).squeeze(1), ty_i)
        opt.zero_grad(); lo.backward(); opt.step()
        if step % 80 == 0 or step == 399:
            with torch.inference_mode():
                pr = probe2(torch.cat([ea, eb, eb - ea], dim=1)).squeeze(1) > 0
                print("    step %3d loss %.4f acc %.1f%%" % (
                    step, float(lo), 100.0 * float((pr.float() == ty_i).float().mean())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
