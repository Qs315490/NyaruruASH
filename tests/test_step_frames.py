"""Per-step screencast logging: the 30 fps frames inside a hold.

The forward-window work needs frames at the resolution the video side already has
(~30 fps over the 0.27 s after a press).  The screenshot stream has them; the
backend used to coalesce it to the newest frame, which threw every one of them
away and left self-play at one frame per control step.

These tests pin the opt-in contract, because the failure mode is quiet: a
recorder that is silently off, or that mixes frames from two steps into one
entry, produces a dataset that still loads and still trains - just against the
wrong window.  What is pinned here is the plumbing, not the decode:

  * off by default, and a normal step logs nothing;
  * on, each dispatched step gets its own entry, in order;
  * `take_step_frames()` returns and clears, and turning recording off clears
    the in-flight step so a later one cannot absorb it.
"""

from __future__ import annotations

import numpy as np

from ash.env.cdp_backend import CdpSpeedrunEnv

GAMEPLAY = {"scene": "Scene_Map", "messageBusy": False, "inGameplay": True,
            "awaitingChoice": False, "confirm": False, "tickerRunning": True}


class _Conn:
    def __init__(self) -> None:
        self.keys: list[str] = []

    def call(self, method, params=None, timeout=None):
        if method == "Input.dispatchKeyEvent" and isinstance(params, dict):
            self.keys.append(str(params.get("type")))
        return None

    def evaluate(self, expr, timeout=None):
        return None

    def close(self):
        return None


def _env() -> CdpSpeedrunEnv:
    env = CdpSpeedrunEnv(auto_connect=False, drive="realtime")
    env.conn = _Conn()  # type: ignore[assignment]
    env.safety = lambda: dict(GAMEPLAY)  # type: ignore[method-assign]
    env.unsafe_reason = lambda: None  # type: ignore[method-assign]
    env.config.frame_ms = 0.0
    return env


def _fake_drain(env: CdpSpeedrunEnv, per_call: int = 1):
    """Stand in for the socket: hand the recorder `per_call` frames per drain."""
    def drain(*, timeout: float = 0.0) -> int:
        for _ in range(per_call):
            env._record_step_frame(np.zeros((4, 4, 3), np.uint8), {"timestamp": 1.23})
        return per_call
    return drain


def test_recording_is_off_by_default() -> None:
    env = _env()
    env._drain_frames = _fake_drain(env)  # type: ignore[method-assign]
    env.step_realtime(0, 0.02)
    assert env.take_step_frames() == []


def test_each_step_gets_its_own_entry_in_order() -> None:
    env = _env()
    env.record_step_frames(True)
    n = [0]

    def drain(*, timeout: float = 0.0) -> int:
        n[0] += 1
        env._record_step_frame(np.full((4, 4, 3), n[0] % 250 + 1, np.uint8),
                               {"timestamp": float(n[0])})
        return 1

    env._drain_frames = drain  # type: ignore[method-assign]
    env.step_realtime(0, 0.02)
    first = env._step_frames[-1]
    env.step_realtime(0, 0.02)
    steps = env.take_step_frames()
    assert len(steps) == 2
    assert len(steps[0]) == len(first) and len(steps[1]) >= 1
    # the two steps hold different frames, so they were not merged or crossed
    assert int(steps[0][0][1][0, 0, 0]) != int(steps[1][0][1][0, 0, 0])
    assert env.take_step_frames() == []


def test_turning_recording_off_clears_the_in_flight_step() -> None:
    env = _env()
    env._drain_frames = _fake_drain(env)  # type: ignore[method-assign]
    env.record_step_frames(True)
    env.step_realtime(0, 0.02)
    assert len(env.take_step_frames()) == 1
    env.record_step_frames(False)
    assert env._step_frames == [] and env._current_step is None
