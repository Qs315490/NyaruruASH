"""Self-play at the right time scale, tested on the videos.

The previous attempt to run this path used the existing self-play sessions, which have one frame
per control step.  Two things were wrong with that, and both are now fixed in the collector:

  * the step was ONE game frame (16.7 ms), not the derived control interval (0.25 s) - the runner
    was built without `frame_skip`;
  * only one frame per step was kept, so a 0.27 s window could not exist at all.

The collector now holds each action for the full 0.25 s and keeps the screencast frames from
inside the hold (~59 fps measured).  This probe trains on that and asks the only question that
matters for the delivery: do self-play labels transfer to the speedrun videos?

Window, on both sides, matches the keycast finding - current frame plus the next 8 at 30 fps
(~0.27 s):

    self-play   tick boundary frame + every other frame of the hold
    video       the tick's 30 fps frame and the 8 after it

Arms, all trained on self-play only: forward window, engine-filtered (intent-consistent), and with
engine displacement as an auxiliary target.  Controls: within one video (temporal 70/30) says
whether the instrument can read keys at all; video -> video is the known cross-pipeline failure.

    uv run python scripts/probe_selfplay30_to_video.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.actions.space import ActionSpace, buttons_from_mask  # noqa: E402
from ash.data.effect import is_effective  # noqa: E402
from ash.utils.device import resolve_device  # noqa: E402

SELFPLAY = sorted((Path("runs") / "selfplay").glob("selfplay-*30fps*.npz"))
VIDEOS = ["BV19s4y1y7un", "BV13HnzzPEEN", "BV1hc411M7GW"]
ACTIONS = ["jump", "attack", "dash", "special", "ult"]

R = 16             # pooled cells per side
WINDOW = 9         # frames: current + 8 future at 30 fps
STEP = 7.5         # video frames per control tick
HALF = 8
EPOCHS = 15
BATCH = 256
LR = 1e-3
AUX_WEIGHT = 0.1
AUX_SCALE = 50.0


def video_path(stem: str, kind: str) -> Path:
    from ash.data.video_pack import artifact
    p = artifact(stem, kind)
    if p is None:
        raise SystemExit("missing %s for %s" % (kind, stem))
    return Path(p)


def keys_of(stems: list[str]) -> list[str]:
    names: set[str] = set()
    for s in stems:
        L = np.load(video_path(s, "labels"), allow_pickle=True)
        names.update(str(n) for n in L["names"])
    return sorted(names)


def pool_gray(frame: np.ndarray) -> np.ndarray:
    H, W = frame.shape[0], frame.shape[1]
    f = H // R
    if frame.ndim == 3:
        frame = frame.mean(axis=2)
    return frame.reshape(R, f, R, f).mean(axis=(1, 3)).reshape(-1).astype(np.float32)


def stack_pool(frames: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([pool_gray(f) for f in frames])


# --------------------------------------------------------------------------- data


def load_selfplay(path: Path, keys: list[str]):
    z = np.load(path, allow_pickle=True)
    frames = z["frames"]
    f30 = z["frames30"]
    acts = np.asarray(z["acts"], np.int64)
    states = [json.loads(s) for s in z["states"]]
    space = ActionSpace.minimal()

    n = min(len(acts), len(f30))
    held = np.zeros((n, len(keys)), np.float32)
    dxy = np.zeros((n, 2), np.float32)
    eff = np.zeros(n, bool)
    for t in range(n):
        mask = int(space.mask_at(int(acts[t])))
        buttons = set(buttons_from_mask(mask))
        for j, nm in enumerate(keys):
            held[t, j] = float(nm in buttons)
        p0 = states[t].get("physics") or {}
        p1 = states[t + 1].get("physics") or {}
        dxy[t, 0] = float(p1.get("px") or 0.0) - float(p0.get("px") or 0.0)
        dxy[t, 1] = float(p1.get("py") or 0.0) - float(p0.get("py") or 0.0)
        eff[t] = is_effective(states[t], states[t + 1], mask)["effective"] is True

    X = np.empty((n, WINDOW * R * R), np.float32)
    for t in range(n):
        seq = [frames[t]]
        sub = f30[t]
        picks = list(range(0, sub.shape[0], 2))[:WINDOW - 1]
        seq += [sub[i] for i in picks]
        while len(seq) < WINDOW:
            seq.append(seq[-1])
        X[t] = stack_pool(seq)
    return X, held, dxy, eff


def load_video(stem: str, keys: list[str]):
    F = np.load(video_path(stem, "frames-30fps"), mmap_mode="r")
    L = np.load(video_path(stem, "labels"), allow_pickle=True)
    held = L["held"].astype(np.float32)
    names = [str(n) for n in L["names"]]
    n_tick = min(len(held), int(len(F) / STEP) - 1)
    centers = (np.arange(n_tick) * STEP).astype(int)
    ok = centers + HALF < len(F)
    centers, held = centers[ok], held[:n_tick][ok]
    Y = np.zeros((len(held), len(keys)), np.float32)
    for j, nm in enumerate(names):
        if nm in keys:
            Y[:, keys.index(nm)] = held[:, j]
    X = np.empty((len(centers), WINDOW * R * R), np.float32)
    for i in range(0, len(centers), 128):
        cc = centers[i:i + 128]
        for k, c in enumerate(cc):
            X[i + k] = stack_pool([np.asarray(F[c + o]) for o in range(HALF + 1)])
    return X, Y


# --------------------------------------------------------------------------- model


class MLP(nn.Module):

    def __init__(self, d_in: int, n_out: int, aux: bool = False) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.GELU())
        self.head = nn.Linear(256, n_out)
        self.aux = nn.Linear(256, 2) if aux else None

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.head(h), (self.aux(h) if self.aux is not None else None)


def train(Xtr, Ytr, dev, *, aux: np.ndarray | None = None, seed: int = 0):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    xt = torch.from_numpy((Xtr - mu) / sd).to(dev)
    yt = torch.from_numpy(Ytr).to(dev)
    at = None if aux is None else torch.from_numpy(aux / AUX_SCALE).to(dev)
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], Ytr.shape[1], aux=aux is not None).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        model.train()
        for _ in range(max(1, len(xt) // BATCH)):
            idx = torch.from_numpy(rng.integers(0, len(xt), size=BATCH)).to(dev)
            logits, delta = model(xt[idx])
            loss = bce(logits, yt[idx])
            if delta is not None:
                loss = loss + AUX_WEIGHT * mse(delta, at[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, mu, sd


def predict(model, mu, sd, X, dev) -> np.ndarray:
    with torch.inference_mode():
        logits, _ = model(torch.from_numpy((X - mu) / sd).to(dev))
        return (logits > 0).float().cpu().numpy()


def scores(P, Y, keys):
    f1s, kj = [], []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if y.sum() < 5:
            continue
        tp = float((P[:, j] * y).sum())
        fp = float((P[:, j] * (1 - y)).sum())
        fn = float(((1 - P[:, j]) * y).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        kj.append(j)
    ones = Y[:, kj] > 0.5
    act = [f1 for f1, j in zip(f1s, kj) if keys[j] in ACTIONS]
    return (float(np.mean(f1s)) if f1s else float("nan"),
            float(np.mean(act)) if act else float("nan"),
            float((P[:, kj][ones] == Y[:, kj][ones]).mean()) if ones.any() else float("nan"),
            float(P[:, kj].mean()),
            {keys[j]: round(f1, 3) for f1, j in zip(f1s, kj)})


def report(tag, P, Y, keys):
    m, a, on, pr, per = scores(P, Y, keys)
    print("   %-18s macroF1 %.3f  action %.3f  ON %.3f  pred %.3f  %s" % (tag, m, a, on, pr, per))
    return m


def main() -> int:
    dev = resolve_device("cuda")
    if not SELFPLAY:
        print("no selfplay-*30fps*.npz under runs/selfplay - run collect_selfplay_30fps.py first")
        return 2
    keys = keys_of(VIDEOS)
    print("device %s | keys %s" % (dev, keys))
    print("self-play sessions: %s" % ", ".join(p.stem for p in SELFPLAY))

    print("\n[self-play] building windows ...")
    sessions = []
    for path in SELFPLAY:
        X, Y, dxy, eff = load_selfplay(path, keys)
        print("  %-24s %s windows, positives %.3f, intent-consistent %.1f%%"
              % (path.stem, X.shape, float(Y.mean()), 100 * float(eff.mean())))
        sessions.append((path.stem, X, Y, dxy, eff))
    Xs = np.concatenate([s[1] for s in sessions])
    Ys = np.concatenate([s[2] for s in sessions])
    ds = np.concatenate([s[3] for s in sessions])
    es = np.concatenate([s[4] for s in sessions])

    print("\n[video] building windows ...")
    vid = {}
    for s in VIDEOS:
        X, Y = load_video(s, keys)
        vid[s] = (X, Y)
        print("  %-14s %s windows, positives %.3f" % (s, X.shape, float(Y.mean())))

    print("\n=== self-play (0.25 s hold + 30 fps window) -> video ===")
    arms = [
        ("future", Xs, Ys, None, None),
        ("future+engine filter", Xs[es], Ys[es], None, None),
        ("future+engine aux", Xs, Ys, ds, None),
    ]
    for name, X, Y, aux, _ in arms:
        model, mu, sd = train(X, Y, dev, aux=aux)
        print("\n%s (train %d windows)" % (name, len(X)))
        for s in VIDEOS:
            report(s, predict(model, mu, sd, vid[s][0], dev), vid[s][1], keys)

    print("\n[in-domain, temporal 70/30 of each self-play session]")
    for name, X, Y, _, _ in sessions:
        cut = int(len(X) * 0.7)
        model, mu, sd = train(X[:cut], Y[:cut], dev)
        report(name, predict(model, mu, sd, X[cut:], dev), Y[cut:], keys)

    print("\n[self-play quantity -> video]  is the failure just too little data?")
    for m in range(1, len(sessions) + 1):
        X = np.concatenate([s[1] for s in sessions[:m]])
        Y = np.concatenate([s[2] for s in sessions[:m]])
        model, mu, sd = train(X, Y, dev)
        ms = [scores(predict(model, mu, sd, vid[s][0], dev), vid[s][1], keys)[0] for s in VIDEOS]
        print("   %d session(s), %5d windows  video macroF1 mean %.3f  (%s)"
              % (m, len(X), float(np.mean(ms)), " ".join("%.3f" % v for v in ms)))

    print("\n[video -> video, same window and instrument]")
    for test in VIDEOS:
        tr = [s for s in VIDEOS if s != test]
        Xtr = np.concatenate([vid[s][0] for s in tr])
        Ytr = np.concatenate([vid[s][1] for s in tr])
        model, mu, sd = train(Xtr, Ytr, dev)
        report("%s -> %s" % ("+".join(x[:6] for x in tr), test[:8]),
               predict(model, mu, sd, vid[test][0], dev), vid[test][1], keys)

    print("\n[within one video, temporal 70/30 - can the window read keys at all?]")
    for s in VIDEOS:
        X, Y = vid[s]
        cut = int(len(X) * 0.7)
        model, mu, sd = train(X[:cut], Y[:cut], dev)
        report(s, predict(model, mu, sd, X[cut:], dev), Y[cut:], keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
