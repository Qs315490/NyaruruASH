"""How much jump does a 250 ms hold actually buy?

The agent's step holds its buttons for exactly one control interval and then
releases - 250 ms at the paper's 0.25 s timestep.  This game has variable-height
jumps: hold longer, jump higher, and the second jump needs a release before the
next press.  If the useful jump needs more than 250 ms, then jump height is not
expressible in the current action encoding at all, and every jump the agent has
ever taken was a short one.  That would also explain a lot of the "the character
cannot get out of here" history, which had been filed under exploration failure.

Measures the apex reached for several hold durations from the same starting spot,
using the engine's own py, and reports it in pixels.

    uv run python scripts/probe_jump_height.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.actions.space import ActionSpace, buttons_from_mask  # noqa: E402
from ash.env.cdp_backend import CdpSpeedrunEnv  # noqa: E402

HOLDS = (2, 4, 8, 15, 25, 40)     # frames at 60 fps: 33 / 67 / 133 / 250 / 417 / 667 ms
TRACK_S = 1.6


def py_of(env) -> float | None:
    p = (env.state().get("physics") or {})
    return float(p["py"]) if p.get("py") is not None else None


def on_ground(env) -> bool:
    a = env.state().get("physics") or {}
    return abs(float(a.get("vy") or 0.0)) < 1e-6


def main() -> int:
    env = CdpSpeedrunEnv(drive="realtime", resize=(128, 128))
    env.connect()
    env.install(seed=0)
    time.sleep(0.5)
    reason = env.unsafe_reason()
    if reason is not None:
        print("拒绝运行：%s" % reason)
        return 3
    st = env.state()
    print("scene=%s map=%s hp=%s physics=%s"
          % (env.safety().get("scene"), (st.get("player") or {}).get("mapId"),
             st.get("hp"), st.get("physics")))

    space = ActionSpace.minimal()
    jump_index = next(i for i, m in enumerate(space.masks) if buttons_from_mask(m) == ("jump",))
    print("jump 是动作 %d\n" % jump_index)

    print("%8s %10s %12s %10s" % ("按住帧数", "≈毫秒", "最高点 py", "相对起跳点"))
    for hold in HOLDS:
        # Settle on the ground before each trial; otherwise the previous jump is
        # still in the air and the measurement is meaningless.
        for _ in range(30):
            if on_ground(env):
                break
            env.step_frame(space.mask_at(0), frames=3)
        base = py_of(env)
        if base is None:
            print("  读不到 physics.py，停止")
            return 3
        env.step_frame(space.mask_at(jump_index), frames=hold, hold=True)
        apex = base
        t0 = time.time()
        while time.time() - t0 < TRACK_S:
            p = py_of(env)
            if p is not None and p < apex:
                apex = p
            time.sleep(0.03)
        print("%8d %10d %12.1f %10.1f" % (hold, int(hold * 1000 / 60), apex, base - apex))

    # Leave the game PAUSED.  `close()` pauses by default, and resuming here
    # was a real bug: the character was left standing in the world after
    # every probe and collection, and the enemies killed it while nobody was
    # driving (measured: hp 150 -> 0 and a GAME OVER screen left sitting).
    # `ash record` resumes because the player is driving and wants the game
    # back; an unattended agent run must not.
    env.close(resume=False)
    print("\n读法：如果「相对起跳点」随按住帧数上升，说明跳跃高度受按住时长控制，"
          "\n而固定 250ms 的编码就只能拿到其中一小截。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
