"""Does filtering self-play by ACTION INTENT improve what the pixels teach?

The gate says 57% of intent-bearing transitions are direction-consistent and 43%
press into a wall, get cancelled by terrain, or are dominated by gravity.  On the
human recordings the equivalent experiment was decisive - informative pairs only
scored macro-recall 0.177 against a 0.135 chance level, the same-size random
subset 0.018 - but that data needed the engine to judge, which was not available.
Now it is, so the same lever can be pulled on the allowed data source.

Sessions are split as whole sessions, not by time inside one: train on two
sessions, test on a third that was collected separately, so no near-duplicate
neighbour can straddle the boundary.

    uv run python scripts/probe_effective_filter.py
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
from ash.data.effect import is_effective  # noqa: E402
from ash.loop.bootstrap import _prep  # noqa: E402
from ash.models.idm import IdmConfig, IdmModel  # noqa: E402
from ash.utils.device import resolve_device  # noqa: E402

SIZE = 128
EPOCHS = 8
BATCH = 8
LR = 3e-4
GRAD_CLIP = 1.0


def load_session(path: str):
    z = np.load(path, allow_pickle=True)
    return (np.asarray(z["frames"]), np.asarray(z["acts"], dtype=np.int64),
            [json.loads(s) for s in z["states"]])


def gate_mask(acts, states, masks) -> np.ndarray:
    """True where the action's intent was actually satisfied."""
    return np.asarray([is_effective(states[t], states[t + 1], masks[int(acts[t])])["effective"] is True
                       for t in range(len(acts))], dtype=bool)


def train_eval(dev, tr, te, label: str) -> dict:
    a_tr, b_tr, y_tr = tr
    a_te, b_te, y_te = te
    torch.manual_seed(0)
    np.random.seed(0)
    model = IdmModel(IdmConfig(image_size=SIZE, num_actions=len(ActionSpace.minimal()))).to(dev)
    head = nn.Linear(model.config.embed_dim * 3, len(ActionSpace.minimal())).to(dev)
    params = list(model.trunk.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    ce = nn.CrossEntropyLoss()

    def forward(idx):
        a = torch.from_numpy(_prep(a_tr[idx], SIZE)).to(dev)
        b = torch.from_numpy(_prep(b_tr[idx], SIZE)).to(dev)
        ea = model.trunk(a.unsqueeze(1)).squeeze(1)
        eb = model.trunk(b.unsqueeze(1)).squeeze(1)
        return head(torch.cat([ea, eb, eb - ea], dim=1))

    order = np.arange(len(y_tr))
    rng = np.random.default_rng(0)
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        total, seen = 0.0, 0
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            loss = ce(forward(idx), torch.from_numpy(y_tr[idx]).to(dev))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            total += float(loss) * len(idx); seen += len(idx)
        print("    [%s] epoch %d train %.4f" % (label, epoch, total / max(1, seen)), flush=True)

    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(y_te), 64):
            a = torch.from_numpy(_prep(a_te[i : i + 64], SIZE)).to(dev)
            b = torch.from_numpy(_prep(b_te[i : i + 64], SIZE)).to(dev)
            ea = model.trunk(a.unsqueeze(1)).squeeze(1)
            eb = model.trunk(b.unsqueeze(1)).squeeze(1)
            preds.append(head(torch.cat([ea, eb, eb - ea], dim=1)).argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    classes = np.unique(y_te)
    recalls = [float((pred[y_te == c] == c).mean()) for c in classes if (y_te == c).any()]
    acc = float((pred == y_te).mean())
    print("    [%s] 留出 %d 对 | 准确率 %.1f%% | macro-recall %.3f（随机 %.3f）| 训练 %d 对"
          % (label, len(y_te), 100 * acc, float(np.mean(recalls)), 1.0 / len(classes),
             len(y_tr)), flush=True)
    return {"acc": acc, "recall": float(np.mean(recalls)), "n_train": len(y_tr)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["runs/selfplay/selfplay-002.npz",
                                                  "runs/selfplay/selfplay-003-outdoor.npz"])
    ap.add_argument("--test", default="runs/selfplay/selfplay-004.npz")
    args = ap.parse_args()
    dev = resolve_device("cuda")
    masks = ActionSpace.minimal().masks

    a_tr, b_tr, y_tr = [], [], []
    for path in args.train:
        frames, acts, states = load_session(path)
        keep = gate_mask(acts, states, masks)
        a_tr.append(frames[:-1][keep]); b_tr.append(frames[1:][keep]); y_tr.append(acts[keep])
        print("训练会话 %s：%d 转移，其中有效 %d（%.1f%%）"
              % (Path(path).name, len(acts), int(keep.sum()), 100 * keep.mean()))
    a_tr = np.concatenate(a_tr); b_tr = np.concatenate(b_tr); y_tr = np.concatenate(y_tr)

    frames, acts, states = load_session(args.test)
    keep_te = gate_mask(acts, states, masks)
    a_te, b_te, y_te = frames[:-1][keep_te], frames[1:][keep_te], acts[keep_te]
    print("留出会话 %s：%d 转移，其中有效 %d（%.1f%%）"
          % (Path(args.test).name, len(acts), int(keep_te.sum()), 100 * keep_te.mean()))

    # The "all" arm needs its own gate-free versions of the same sessions.
    a_all, b_all, y_all = [], [], []
    for path in args.train:
        f, act, _ = load_session(path)
        a_all.append(f[:-1]); b_all.append(f[1:]); y_all.append(act)
    a_all = np.concatenate(a_all); b_all = np.concatenate(b_all); y_all = np.concatenate(y_all)

    rng = np.random.default_rng(0)
    n_eff = len(y_tr)
    rand = np.zeros(len(y_all), dtype=bool)
    rand[rng.permutation(len(y_all))[:n_eff]] = True

    print("\n评估集 = 留出会话里**意图一致**的 %d 对" % len(y_te))
    results = {
        "A 全部转移": train_eval(dev, (a_all, b_all, y_all),
                                 (a_te, b_te, y_te), "A"),
        "B 只喂意图一致": train_eval(dev, (a_tr, b_tr, y_tr),
                                     (a_te, b_te, y_te), "B"),
        "C 随机等量": train_eval(dev, (a_all[rand], b_all[rand], y_all[rand]),
                                 (a_te, b_te, y_te), "C"),
    }
    print("\n=== 汇总（留出会话的意图一致转移）===")
    print("%-18s %8s %10s %14s" % ("臂", "训练对数", "准确率", "macro-recall"))
    for name, r in results.items():
        print("%-18s %8d %9.1f%% %14.3f" % (name, r["n_train"], 100 * r["acc"], r["recall"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
