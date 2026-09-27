"""Would a latent action model work on THIS data?  Three judgements, one verdict.

The appeal of the LAM family (Genie, LAPO, CLAM) for this project is that it does
not need action labels at all, which is exactly what "learn from videos, no human
data" asks for.  It also sidesteps the label-identifiability failure measured
here: 35% of human pairs and 43% of self-play intent-bearing pairs cannot teach a
key label because the action had no visible consequence.

What it cannot sidestep is the other half, and this project has already measured
it: with no input at all the picture changes MOST (gravity, 0.0998 against 0.0039
for actually walking), and the global picture-change magnitude correlates with
the player's own displacement at -0.011.  A latent trained to make the next frame
predictable will happily spend its capacity on whatever changes the picture -
which here is the world, not the input.  From pixels alone, "the world moved" and
"I moved" are the same observation.

So a LAM is not judged by reconstruction quality; it is judged by whether its
latents mean an action.  The engine teacher - allowed at training time, and
available to almost nobody doing this work - is the judge:

  1. action alignment   can the true action be read off the latent at all?
  2. CONFOUND TEST      on transitions where the player did NOT move and the
                        world did, is the latent constant?  Varying latents there
                        mean the model is encoding scenery motion, and no amount
                        of scale fixes that on this data.
  3. cross-session      does the same action land on the same latent in a session
                        the model never trained on?

    uv run python scripts/probe_latent_action.py --codes 16 --epochs 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.utils.device import resolve_device  # noqa: E402

SIZE = 64          # latents only need to explain the change; 64 keeps this cheap
BATCH = 16
LR = 3e-4
CHANGE_THRESHOLD = 0.005   # the recorder's own "the picture moved" cut


def load(path: str):
    z = np.load(path, allow_pickle=True)
    frames = np.asarray(z["frames"], dtype=np.uint8)
    acts = np.asarray(z["acts"], dtype=np.int64)
    states = [json.loads(s) for s in z["states"]]
    return frames, acts, states


def player_moved(states) -> np.ndarray:
    """The teacher's own verdict, straight from the physics plugin."""
    out = []
    for t in range(len(states) - 1):
        a = (states[t] or {}).get("physics") or {}
        b = (states[t + 1] or {}).get("physics") or {}
        if not a or not b:
            out.append(None)
            continue
        moved = any(a.get(k) is not None and b.get(k) is not None and float(a[k]) != float(b[k])
                    for k in ("px", "py", "vx", "vy"))
        out.append(bool(moved))
    return np.asarray(out, dtype=object)


def down(frames: np.ndarray) -> np.ndarray:
    import cv2

    return np.stack([cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for f in frames])


def tensor(x, dev):
    return torch.from_numpy(np.ascontiguousarray(x)).to(dev).permute(0, 3, 1, 2).float().div_(255.0)


class LatentActionModel(nn.Module):
    """q(z | obs_t, obs_t+1) with a codebook, and p(obs_t+1 | obs_t, z).

    Deliberately small.  The question is not whether it reconstructs well - a
    world model of this game is a different project - but whether the latent it
    discovers corresponds to an action, and that shows up at any size.
    """

    def __init__(self, codes: int) -> None:
        super().__init__()
        self.codes = codes
        self.enc = nn.Sequential(
            nn.Conv2d(6, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_codes = nn.Linear(64, codes)
        #: VQ codebook.  A soft codebook plus an entropy bonus has no pressure to
        #: carry anything: measured, swapping z between samples changed the
        #: reconstruction loss by -0.0%, i.e. the decoder ignored its latent and
        #: copied obs_t.  The trivial copy is nearly optimal here because walking
        #: changes the picture by 0.0039 mean absolute difference, so the latent
        #: has to be forced to exist.
        self.book = nn.Parameter(torch.randn(codes, 64) * 0.1)
        self.from_code = nn.Linear(64, 64 * (SIZE // 8) * (SIZE // 8))
        # 8x8 -> 16 -> 32 -> 64, so the output matches SIZE; three x4 steps gave
        # 512 and the residual add then failed on a shape mismatch.
        self.dec = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 3, 3, 1, 1),
        )

    def forward(self, a, b):
        """Encoder -> nearest code -> decoder predicts the CHANGE from the code."""
        z_e = self.enc(torch.cat([a, b], dim=1)).flatten(1)        # (B, D)
        dist = torch.cdist(z_e, self.book)                          # (B, K)
        idx = dist.argmin(dim=1)                                    # (B,)
        z_q = self.book[idx]                                        # (B, D)
        # Straight-through: the decoder sees the quantised vector, the encoder
        # still gets a gradient.
        z_st = z_e + (z_q - z_e).detach()
        h = self.from_code(z_st).view(-1, 64, SIZE // 8, SIZE // 8)
        return z_e, z_q, z_st, idx, self.dec(h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["runs/selfplay/selfplay-002.npz",
                                                  "runs/selfplay/selfplay-003-outdoor.npz"])
    ap.add_argument("--test", default="runs/selfplay/selfplay-004.npz")
    ap.add_argument("--codes", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--min-motion", type=float, default=CHANGE_THRESHOLD,
                    help="训练只用画面变化超过它的对；0 = 全部")
    args = ap.parse_args()
    dev = resolve_device("cuda")

    tr_a, tr_b, tr_y, tr_moved, tr_motion = [], [], [], [], []
    for path in args.train:
        frames, acts, states = load(path)
        f = down(frames)
        tr_a.append(f[:-1]); tr_b.append(f[1:]); tr_y.append(acts)
        tr_moved.append(player_moved(states))
        d = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
        tr_motion.append(d.mean(axis=(1, 2, 3)) / 255.0)
    a_tr = np.concatenate(tr_a); b_tr = np.concatenate(tr_b)
    y_tr = np.concatenate(tr_y); moved_tr = np.concatenate(tr_moved)
    motion_tr = np.concatenate(tr_motion)
    # A latent action has to have something to explain.  Without this filter the
    # measured shuffle gap was +0.0%: in an interior room the frame-to-frame
    # change is so small that predicting a constant is already optimal, so the
    # latent is redundant.  Restricting training to pairs where the picture
    # actually changed is a data fix, not a bigger model.
    keep = motion_tr >= args.min_motion
    print("训练会话共 %d 对；画面真的变了 %d 对（%.1f%%）用于训练（阈值 %.4f）"
          % (len(y_tr), int(keep.sum()), 100 * keep.mean(), args.min_motion))
    if keep.sum() < 200:
        print("可用对太少（%d），不做结论。" % int(keep.sum()))
        return 2
    a_tr, b_tr, y_tr = a_tr[keep], b_tr[keep], y_tr[keep]
    moved_tr, motion_tr = moved_tr[keep], motion_tr[keep]
    print("训练 %d 对（%d 个会话）| 留出 %s" % (len(y_tr), len(args.train), Path(args.test).name))

    frames, acts, states = load(args.test)
    f = down(frames)
    a_te, b_te, y_te = f[:-1], f[1:], acts
    moved_te = player_moved(states)
    d = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    motion_te = d.mean(axis=(1, 2, 3)) / 255.0

    torch.manual_seed(0)
    model = LatentActionModel(args.codes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    order = np.arange(len(y_tr))
    rng = np.random.default_rng(0)
    for epoch in range(args.epochs):
        rng.shuffle(order)
        total, seen = 0.0, 0
        model.train()
        for i in range(0, len(order), BATCH):
            idx_b = order[i : i + BATCH]
            at = tensor(a_tr[idx_b], dev); bt = tensor(b_tr[idx_b], dev)
            z_e, z_q, z_st, _, pred = model(at, bt)
            target = bt - F.interpolate(at, size=(SIZE, SIZE), mode="nearest")
            rec = F.mse_loss(pred, target)
            commit = 0.25 * F.mse_loss(z_e, z_q.detach())
            book = 0.25 * F.mse_loss(z_q, z_e.detach())
            loss = rec + commit + book
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(rec) * len(idx_b); seen += len(idx_b)
        print("  epoch %d 变化图 MSE %.5f" % (epoch, total / max(1, seen)), flush=True)

    model.eval()
    def codes_of(a, b):
        """The discrete code each transition was assigned to."""
        out = []
        with torch.inference_mode():
            for i in range(0, len(a), 128):
                _, _, _, c, _ = model(tensor(a[i : i + 128], dev), tensor(b[i : i + 128], dev))
                out.append(c.cpu().numpy())
        return np.concatenate(out)

    # 关键对照：把 z 换成别的样本的 z，重建若不变差，说明解码器根本没用 z，
    # 那么三个判据全是空的——报一个「模型没训起来」的负结果比不报更糟。
    with torch.inference_mode():
        mse_true, mse_shuffled, codes = [], [], []
        for i in range(0, len(a_te), 128):
            at = tensor(a_te[i : i + 128], dev); bt = tensor(b_te[i : i + 128], dev)
            _, _, _, idx_b, pred = model(at, bt)
            tgt = bt - F.interpolate(at, size=(SIZE, SIZE), mode="nearest")
            mse_true.append(float(F.mse_loss(pred, tgt)))
            perm = torch.randperm(len(idx_b), device=dev)
            h = model.from_code(model.book[idx_b[perm]]).view(-1, 64, SIZE // 8, SIZE // 8)
            mse_shuffled.append(float(F.mse_loss(model.dec(h), tgt)))
            codes.append(idx_b.cpu().numpy())
    mse_true = float(np.mean(mse_true)); mse_shuffled = float(np.mean(mse_shuffled))
    gap = 100 * (mse_shuffled - mse_true) / max(1e-9, mse_true)
    print("\n  z 使用量对照：真码重建 MSE %.5f | 打乱码 %.5f | 打乱后差 %+.1f%%" %
          (mse_true, mse_shuffled, gap))
    idx_te = np.concatenate(codes)
    if gap < 5.0:
        # Not a verdict about latent action models - a verdict about this run.
        # Reporting the three judgements anyway would dress up an unused latent
        # as evidence that the approach fails on this data.
        print("  ⇒ 打乱 latent 不使重建变差：**模型没在用 latent**，三个判据都无从谈起。")
        print("     这是本次运行的结论，不是「LAM 在这份数据上不行」的结论。")
        return 2
    idx_tr = codes_of(a_tr, b_tr)
    used = len(np.unique(idx_tr))
    print("\n码本：%d 个码，训练集用到 %d 个" % (args.codes, used))

    # ---- 判据 1：动作能不能从 latent 读出来（用老师之外的、留出会话上的小分类器）
    from sklearn.linear_model import LogisticRegression

    def onehot(c, k):
        m = np.zeros((len(c), k), np.float32)
        m[np.arange(len(c)), c] = 1.0
        return m

    clf = LogisticRegression(max_iter=500)
    clf.fit(onehot(idx_tr, args.codes), y_tr)
    pred = clf.predict(onehot(idx_te, args.codes))
    classes = np.unique(y_te)
    recalls = [float((pred[y_te == c] == c).mean()) for c in classes if (y_te == c).any()]
    base = float(np.bincount(y_te).max() / len(y_te))
    print("\n判据1 动作对齐：留出会话准确率 %.1f%%（基线 %.1f%%，随机 %.1f%%）| macro-recall %.3f"
          % (100 * (pred == y_te).mean(), 100 * base, 100 / len(classes), float(np.mean(recalls))))

    # ---- 判据 2（生死线）：玩家没动、世界在动的转移上，latent 是否稳定
    still_world = np.asarray([(m is False and mo >= CHANGE_THRESHOLD)
                              for m, mo in zip(moved_te, motion_te)], dtype=bool)
    player_here = np.asarray([(m is True) for m in moved_te], dtype=bool)

    def concentration(sel, name):
        if sel.sum() < 30:
            print("  %-26s 样本太少（%d）" % (name, int(sel.sum())))
            return None
        counts = np.bincount(idx_te[sel], minlength=args.codes).astype(float)
        p = counts / counts.sum()
        nz = p[p > 0]
        ent = float(-(nz * np.log(nz)).sum() / np.log(args.codes))
        print("  %-26s n=%5d | 用到 %2d 个码 | 归一化熵 %.3f | 最大码占比 %.1f%%"
              % (name, int(sel.sum()), int((p > 0).sum()), ent, 100 * p.max()))
        return ent

    print("\n判据2 混淆测试（生死线）：玩家没动但画面在动 = 世界自己在动")
    ent_world = concentration(still_world, "世界动/玩家不动")
    ent_player = concentration(player_here, "玩家自己在动")
    if ent_world is not None and ent_player is not None:
        print("  ⇒ 两者熵接近（或世界动那边更高）说明 latent 主要在编码世界运动，**这条死**；")
        print("     世界动那边的熵显著更低说明 latent 归于「玩家自己的动作」，可以继续。")

    # ---- 判据 3：同一个码在留出会话里是否还对应同一个动作
    maj = {}
    for c in np.unique(idx_tr):
        vals, cnt = np.unique(y_tr[idx_tr == c], return_counts=True)
        maj[int(c)] = int(vals[cnt.argmax()])
    mapped = np.asarray([maj.get(int(c), 0) for c in idx_te])
    classes_te = np.unique(y_te)
    r3 = [float((mapped[y_te == c] == c).mean()) for c in classes_te if (y_te == c).any()]
    print("\n判据3 跨会话一致：码→多数动作（训练集上统计）在留出会话上 %.1f%%（基线 %.1f%%）| macro-recall %.3f"
          % (100 * (mapped == y_te).mean(), 100 * base, float(np.mean(r3))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
