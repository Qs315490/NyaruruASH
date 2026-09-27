"""Collect self-play data with the engine teacher attached.

This is the allowed data source: the agent's own actions, labelled by the agent
itself, with the game's own readout of the player alongside each step.  No human
input is involved anywhere.

It exists to answer one question before anything is trained: how much of
self-play data is INFORMATIVE?  On the human recordings the answer was 35%
unidentifiable, and feeding only the informative pairs beat feeding everything
(macro-recall 0.177 against a 0.135 chance level, versus 0.018 for a random
subset of the same size).  Self-play may be a different number, and the filter
should be chosen from it rather than assumed.

Writes `runs/selfplay/<name>.npz`:
    frames   (T, 128, 128, 3) uint8     observations at the model size
    acts     (T-1,)            int64    the action taken at t, as a class index
    motion   (T-1,)            float32  mean |frame[t+1] - frame[t]| / 255
    moved    (T-1,)            bool     the PLAYER's own body moved (the teacher)
    known    (T-1,)            bool     the teacher could be read at all
    states   (T,)              object   the raw per-step engine readout

    uv run python scripts/collect_selfplay.py --steps 600 --name selfplay-001
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
from ash.env.cdp_backend import CdpSpeedrunEnv  # noqa: E402
from ash.models.ash_policy import AshPolicy, AshPolicyConfig  # noqa: E402
from ash.loop.runner import InferenceRunner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600, help="随机步数（每步 %ds 游戏时间）")
    ap.add_argument("--name", default="selfplay-001")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--out", default="runs/selfplay")
    ap.add_argument("--hold-actions", action="store_true",
                    help="按住跨步（跳跃高度与二段跳需要它）")
    ap.add_argument("--motifs", action="store_true",
                    help="用短动作序列探索：均匀随机撞不到二段跳（1/400）")
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
        # A stopped ticker means the game does not advance while keys are held, so
        # the run would collect a frozen character and call it data.
        print("拒绝采集：引擎 ticker 未运行（游戏处于暂停态），先在页面里 __ash.pump.resume()")
        return 3
    print("scene=%s map=%s hp=%s | 护栏允许" % (
        safety.get("scene"), (env.state().get("player") or {}).get("mapId"),
        env.state().get("hp")))

    # The runner is used for its random rollout, which is the tested path through
    # the scene guardrails; the policy and K play no part in it.
    # A throwaway policy: the random rollout never consults it, but the runner's
    # constructor insists on one.  Built tiny so this costs nothing.
    space = env.action_space
    policy = AshPolicy(AshPolicyConfig(image_size=args.image_size, w_s=1, w_l=1,
                                       num_layers=1, num_actions=len(space)))
    runner = InferenceRunner(lambda: env, policy, None, action_space=space,
                             w_s=1, w_l=1, image_size=args.image_size, device="cpu",
                             hold_actions=bool(args.hold_actions))
    # Coverage motifs.  Uniform sampling reaches jump,jump about once in twenty
    # actions, but jump,noop,jump - the double jump - about once in four hundred,
    # so a round would contain a handful.  These are action sequences, still
    # executed by the agent's own input; the point is that the dynamics get seen.
    motifs: list[list[int]] = []
    if args.motifs:
        def idx(name: str) -> int:
            # The noop mask has an EMPTY button set, so its name here is "" and
            # not "noop": looking for the literal string raised StopIteration and
            # killed the collection before a single step ran.
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

    started = time.time()
    traj = runner.random_rollout(env, args.steps, args.seed, motifs=motifs or None)
    print("采到 %d 步（%.1f s 墙钟）" % (len(traj["act"]), time.time() - started))

    frames = np.asarray(traj["obs"], dtype=np.uint8)
    acts = np.asarray(traj["act"], dtype=np.int64)
    states = traj["state"]
    motion = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    motion = (motion.mean(axis=(1, 2, 3)) / 255.0).astype(np.float32)

    report = describe_effects([traj], motion=motion, still_threshold=0.005)
    per_step = report.pop("per_step")
    moved = np.asarray([p["moved"] for p in per_step], dtype=bool)
    known = np.asarray([p["known"] for p in per_step], dtype=bool)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("%s.npz" % args.name)
    np.savez_compressed(
        path, frames=frames, acts=acts, motion=motion, moved=moved, known=known,
        states=np.asarray([json.dumps(s, ensure_ascii=False) for s in states], dtype=object),
    )

    print("\n=== 这批自博弈数据是什么构成的 ===")
    for key in ("transitions", "own_moved", "own_still", "unknown_state",
                "screen_changed", "screen_still", "nothing_changed", "noop_label"):
        print("  %-16s %6d" % (key, report[key]))
    print("  %-16s %s" % ("有信息比例", "%.1f%%" % (100 * (report["own_moved"] / max(1, report["transitions"])))))
    print("  %-16s %.1f%%" % ("人动/画面也动", 100 * report["share_own_moved"]))
    print("  %-16s %s" % ("完全无变化", "%.1f%%" % (100 * (report["share_nothing_changed"] or 0))))
    print("\n写入 %s（%.1f MB）" % (path, path.stat().st_size / 1e6))
    # Leave the game PAUSED.  `close()` pauses by default, and resuming here
    # was a real bug: the character was left standing in the world after
    # every probe and collection, and the enemies killed it while nobody was
    # driving (measured: hp 150 -> 0 and a GAME OVER screen left sitting).
    # `ash record` resumes because the player is driving and wants the game
    # back; an unattended agent run must not.
    env.close(resume=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
