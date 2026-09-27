"""Does an action show up in the engine telemetry at all?

The teacher is only worth using if a key press is visible in it.  Standing still
proved `physics.px/py` reads (240, 455) and reports "not moved"; that says
nothing about whether pressing left moves it.  This is the smallest test of the
claim, and it runs before any multi-hour self-play: a handful of steps per
action, interleaved so the character does not drift into a wall.

It is a probe, so it presses keys - the guardrails stay on, which means it only
runs inside Scene_Map and stops on any scene it does not know.

    uv run python scripts/probe_telemetry_semantics.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.actions.space import ActionSpace, buttons_from_mask  # noqa: E402
from ash.env.cdp_backend import CdpSpeedrunEnv  # noqa: E402

TRIALS = 5
FRAMES = 15  # one control interval at control_interval_s = 0.25


def _physics(state):
    return (state or {}).get("physics") or {}


def _screen(state):
    p = (state or {}).get("player") or {}
    return p.get("screenX"), p.get("screenY")


def main() -> int:
    env = CdpSpeedrunEnv(drive="realtime", resize=(128, 128))
    env.connect()
    env.install(seed=0)
    time.sleep(0.5)
    reason = env.unsafe_reason()
    if reason is not None:
        print("拒绝运行：%s" % reason)
        return 3
    print("scene=%s | 场景护栏允许派发按键" % (env.safety().get("scene"),))

    space = ActionSpace.minimal()
    by_name = {}
    for i, mask in enumerate(space.masks):
        name = ", ".join(buttons_from_mask(mask)) or "noop"
        if name in ("noop", "left", "right", "jump"):
            by_name[name] = i

    results: dict[str, dict[str, list[float]]] = {}
    for _ in range(TRIALS):
        for name, index in by_name.items():
            before = env.state()
            frame_before = env.capture_once()
            env.step(space.mask_at(index), frames=FRAMES, hold=True)
            after = env.state()
            frame_after = env.capture_once()

            p0, p1 = _physics(before), _physics(after)
            s0, s1 = _screen(before), _screen(after)
            motion = float(np.abs(frame_after.astype(np.int16)
                                  - frame_before.astype(np.int16)).mean() / 255.0)
            r = results.setdefault(name, {"dpx": [], "dpy": [], "dscreen": [], "motion": [],
                                          "vx": []})
            r["dpx"].append(float((p1.get("px") or 0) - (p0.get("px") or 0)))
            r["dpy"].append(float((p1.get("py") or 0) - (p0.get("py") or 0)))
            r["dscreen"].append(abs(float((s1[0] or 0) - (s0[0] or 0)))
                                + abs(float((s1[1] or 0) - (s0[1] or 0))))
            r["motion"].append(motion)
            r["vx"].append(float(p1.get("vx") or 0))

    print("\n每个动作 %d 次，每次按 %d 帧（= %.2f s 游戏时间）"
          % (TRIALS, FRAMES, FRAMES / 60.0))
    print("%-7s %18s %18s %10s %10s" % ("动作", "physics Δpx 中位", "physics Δpy 中位",
                                        "屏幕Δ中位", "画面变化"))
    for name, r in results.items():
        print("%-7s %18.1f %18.1f %10.1f %10.4f"
              % (name, statistics.median(r["dpx"]), statistics.median(r["dpy"]),
                 statistics.median(r["dscreen"]), statistics.median(r["motion"])))
    print("\n动过的比例（|Δpx|>0）：%s"
          % {n: "%.0f%%" % (100 * np.mean(np.abs(r["dpx"]) > 0))
             for n, r in results.items()})
    env.close(resume=True)
    print("游戏已留在暂停态（__ash.pump.resume() 可恢复）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
