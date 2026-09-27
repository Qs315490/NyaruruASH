"""Does holding across steps actually buy the jump, and the second jump?

`probe_jump_height.py` measured the mechanic: 250 ms of hold reaches 172 px where
the game allows 212, and the double jump needs a release at the apex.  The
held-input mode now makes both expressible - repeating an action lengthens the
hold, switching action releases - so this measures whether that is true on the
real game rather than only in the unit tests.

Patterns are sequences of action indices; each is held for one control interval,
and the held-input mode keeps the buttons down between identical ones.

    uv run python scripts/probe_jump_hold.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.actions.space import ActionSpace, buttons_from_mask  # noqa: E402
from ash.env.cdp_backend import CdpSpeedrunEnv  # noqa: E402

JUMP_FRAMES = 15          # one control interval
TRACK_S = 2.5
PATTERNS = [
    ("jump", [("jump",)]),
    ("jump×2（按住两步）", [("jump",), ("jump",)]),
    ("jump×3（按住三步）", [("jump",), ("jump",), ("jump",)]),
    ("jump→noop→jump（最高点松开再按）", [("jump",), (), ("jump",)]),
    ("jump→noop→noop→jump", [("jump",), (), (), ("jump",)]),
    ("jump×2→noop→jump", [("jump",), ("jump",), (), ("jump",)]),
]


def main() -> int:
    env = CdpSpeedrunEnv(drive="realtime", resize=(128, 128))
    env.connect()
    env.install(seed=0)
    time.sleep(0.5)
    reason = env.unsafe_reason()
    if reason is not None:
        print("拒绝运行：%s" % reason)
        return 3
    space = ActionSpace.minimal()
    index_of = {}
    for i, mask in enumerate(space.masks):
        index_of[", ".join(buttons_from_mask(mask)) or "noop"] = i
    st = env.state()
    print("scene=%s map=%s hp=%s physics=%s" % (
        env.safety().get("scene"), (st.get("player") or {}).get("mapId"),
        st.get("hp"), st.get("physics")))

    def py() -> float | None:
        p = env.state().get("physics") or {}
        return float(p["py"]) if p.get("py") is not None else None

    def settle() -> None:
        for _ in range(40):
            p = env.state().get("physics") or {}
            if abs(float(p.get("vy") or 0.0)) < 1e-6:
                return
            env.step_frame(space.noop, frames=3)

    print("\n%34s %10s %12s" % ("模式", "最高点", "相对起跳"))
    try:
        for label, pattern in PATTERNS:
            settle()
            base = py()
            if base is None:
                print("读不到 physics.py")
                return 3
            for j, buttons in enumerate(pattern):
                name = ", ".join(buttons) or "noop"
                mask = space.mask_at(index_of[name])
                last = j == len(pattern) - 1
                if last:
                    # Release after the final element so the character can land.
                    env.step_holding(mask, frames=JUMP_FRAMES)
                    env.release()
                else:
                    env.step_holding(mask, frames=JUMP_FRAMES)
            apex = base
            t0 = time.time()
            while time.time() - t0 < TRACK_S:
                p = py()
                if p is not None and p < apex:
                    apex = p
                time.sleep(0.03)
            print("%34s %10.1f %12.1f" % (label, apex, base - apex))
    finally:
        env.release()
    # Leave the game PAUSED.  `close()` pauses by default, and resuming here
    # was a real bug: the character was left standing in the world after
    # every probe and collection, and the enemies killed it while nobody was
    # driving (measured: hp 150 -> 0 and a GAME OVER screen left sitting).
    # `ash record` resumes because the player is driving and wants the game
    # back; an unattended agent run must not.
        env.close(resume=False)
    print("\n（单次 250ms 按住 = 172 px，最大 = 212 px，来自 probe_jump_height.py）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
