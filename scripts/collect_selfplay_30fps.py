"""Self-play with the 30 fps frames INSIDE each 0.25 s hold.

The forward-window finding (action keys 0.126 -> 0.678 within a video, peak at 0.27 s) was
measured on 30 fps video frames.  Self-play was collected one frame per control step - 4 fps - so
it cannot represent a 0.27 s window at all, and the earlier "self-play + forward window" attempt
was forced to use future TICKS (0.75 s) as a substitute.  It does not need to be:

    Page.startScreencast is already running on this connection.  The backend used to coalesce it
    to its newest frame, but while the buttons are held the frames of that interval - the actual
    consequence of the press - are exactly what the screenshot stream contains.

So the backend gained an opt-in per-step log (`record_step_frames` / `take_step_frames`) and this
collector uses it.  Everything else is the tested path: the same runner random rollout, the same
scene guardrails, the same engine-state teacher, no human input anywhere.

Writes `runs/selfplay/<name>.npz`:
    frames        (T, S, S, 3) uint8     the per-step observation (tick boundaries, as before)
    frames30      (T-1,) object          per step: (n_t, S, S, 3) uint8 frames during the hold
    frames30_t    (T-1,) object          per step: the screencast timestamps of those frames
    acts          (T-1,) int64           the action taken at t, as a class index
    motion/moved/known/states            unchanged from `collect_selfplay`

    uv run python scripts/collect_selfplay_30fps.py --steps 600 --motifs --hold-actions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.data.effect import describe_effects  # noqa: E402
from ash.actions.space import buttons_from_mask  # noqa: E402
from ash.config import load_game_config  # noqa: E402
from ash.env.cdp_backend import CdpSpeedrunEnv  # noqa: E402
from ash.models.ash_policy import AshPolicy, AshPolicyConfig  # noqa: E402
from ash.loop.runner import InferenceRunner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--name", default="selfplay-30fps-001")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--out", default="runs/selfplay")
    ap.add_argument("--hold-actions", action="store_true",
                    help="按住跨步；30fps 窗正好覆盖一次按住（跳跃高度需要它）")
    ap.add_argument("--motifs", action="store_true")
    args = ap.parse_args()

    env = CdpSpeedrunEnv(drive="realtime", resize=(args.image_size, args.image_size))
    env.connect()
    env.install(seed=0)
    time.sleep(0.6)

    safety = env.safety() or {}
    reason = env.unsafe_reason()
    if reason is not None:
        print("拒绝采集：%s" % reason)
        return 3
    if not safety.get("tickerRunning"):
        print("拒绝采集：引擎 ticker 未运行（游戏处于暂停态），先在页面里 __ash.pump.resume()")
        return 3
    print("scene=%s map=%s hp=%s | 护栏允许" % (
        safety.get("scene"), (env.state().get("player") or {}).get("mapId"),
        env.state().get("hp")))

    space = env.action_space
    game = load_game_config()
    # The agent's timestep, from the single source (game.yaml:control_interval_s).
    # Leaving it at the constructor default of 1 means one game frame (16.7 ms)
    # per action, which is 15x finer than the 4 fps corpus the IDM is applied to.
    frame_skip = game.control_frame_skip
    policy = AshPolicy(AshPolicyConfig(image_size=args.image_size, w_s=1, w_l=1,
                                       num_layers=1, num_actions=len(space)))
    runner = InferenceRunner(lambda: env, policy, None, action_space=space,
                             w_s=1, w_l=1, image_size=args.image_size, device="cpu",
                             frame_skip=frame_skip, hold_actions=bool(args.hold_actions))
    print("control: %.3f s/step = %d game frames (corpus %.1f fps)"
          % (game.control_interval_s, frame_skip, game.corpus_fps))
    motifs: list[list[int]] = []
    if args.motifs:
        def idx(name: str) -> int:
            want = "" if name == "noop" else name
            return next(i for i, m in enumerate(space.masks)
                        if ", ".join(buttons_from_mask(m)) == want)

        named = [
            ("jump",), ("jump", "jump"), ("jump", "jump", "jump"),
            ("jump", "noop", "jump"), ("jump", "noop", "jump", "noop", "jump"),
            ("right",), ("right", "right"), ("right", "right", "right"),
            ("left",), ("left", "left"),
            ("right", "jump"), ("right", "jump", "jump"),
            ("right", "jump", "noop", "jump"),
            ("dash",), ("right", "dash"),
        ]
        motifs = [[idx(n) for n in seq] for seq in named]
        print("motifs: %d 种，最长 %d 步" % (len(motifs), max(len(m) for m in motifs)))

    env.record_step_frames(True)
    started = time.time()
    traj = runner.random_rollout(env, args.steps, args.seed, motifs=motifs or None)
    wall = time.time() - started
    step_frames = env.take_step_frames()
    env.record_step_frames(False)
    print("采到 %d 步（%.1f s 墙钟）；记录到 %d 段按住帧" % (len(traj["act"]), wall, len(step_frames)))

    counts = np.asarray([len(s) for s in step_frames], dtype=np.int32)
    if len(step_frames):
        print("每步帧数：中位 %d，最小 %d，最大 %d，合计 %d"
              % (int(np.median(counts)), int(counts.min()), int(counts.max()), int(counts.sum())))
        all_t = [t for s in step_frames for t, _ in s]
        dt = np.diff(all_t)
        if len(dt) and dt.size:
            print("帧间隔中位 %.1f ms（≈%.1f fps）" % (1000 * float(np.median(dt)),
                                                    1.0 / max(1e-6, float(np.median(dt)))))
    if len(step_frames) != len(traj["act"]):
        print("警告：帧段数 %d != 动作数 %d（护栏跳过的步不会产生帧段，需对齐）"
              % (len(step_frames), len(traj["act"])))

    frames = np.asarray(traj["obs"], dtype=np.uint8)
    acts = np.asarray(traj["act"], dtype=np.int64)
    states = traj["state"]
    motion = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    motion = (motion.mean(axis=(1, 2, 3)) / 255.0).astype(np.float32)

    report = describe_effects([traj], motion=motion, still_threshold=0.005)
    per_step = report.pop("per_step")
    moved = np.asarray([p["moved"] for p in per_step], dtype=bool)
    known = np.asarray([p["known"] for p in per_step], dtype=bool)

    f30 = np.empty(len(step_frames), dtype=object)
    t30 = np.empty(len(step_frames), dtype=object)
    for i, seq in enumerate(step_frames):
        f30[i] = np.stack([f for _, f in seq]) if seq else np.zeros((0, args.image_size,
                                                                    args.image_size, 3), np.uint8)
        t30[i] = np.asarray([t for t, _ in seq], np.float64)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("%s.npz" % args.name)
    np.savez_compressed(
        path, frames=frames, acts=acts, motion=motion, moved=moved, known=known,
        frames30=f30, frames30_t=t30, frames30_n=counts,
        states=np.asarray([json.dumps(s, ensure_ascii=False) for s in states], dtype=object),
    )

    print("\n=== 这批自博弈数据是什么构成的 ===")
    for key in ("transitions", "own_moved", "own_still", "unknown_state",
                "screen_changed", "screen_still", "nothing_changed", "noop_label"):
        print("  %-16s %6d" % (key, report[key]))
    print("  %-16s %s" % ("有信息比例", "%.1f%%" % (100 * (report["own_moved"] / max(1, report["transitions"])))))
    print("\n写入 %s（%.1f MB）" % (path, path.stat().st_size / 1e6))
    env.close(resume=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
