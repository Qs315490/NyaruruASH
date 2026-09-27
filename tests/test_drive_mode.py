"""Self-play must not take the engine's frame loop away from it.

The frame pump drives the ticker by hand with a synthetic clock.  That is what
deterministic replay and search need, but it CHANGES the game.  Measured on this
game, standing on a damage trap:

    real ticker : _pRealState cycles 3-6-1-0-9-4-7..., and the trap teleports
                  the character out (map 8 (0,8) -> map 14 (0,18))
    frame pump  : _pRealState pinned at 6 for all 1200 pumped frames, and the
                  trap never resolves

So a pumped self-play run gathers trajectories of a character that cannot move -
which is upstream of every "the IDM did not learn" symptom that followed.

`drive="realtime"` keeps the engine's loop and only dispatches input on a
real-time schedule.  These tests pin that the two modes stay distinct and that
realtime really leaves the ticker alone.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from ash.actions.space import ActionSpace  # noqa: E402
from ash.env.cdp_backend import CdpSpeedrunEnv  # noqa: E402

GAMEPLAY = {"scene": "Scene_Map", "messageBusy": False, "inGameplay": True,
            "awaitingChoice": False, "confirm": False, "tickerRunning": True}

#: The exact probe expressions, so assertions cannot be satisfied by the
#: agent.js source text that also gets evaluated.
PUMP_INSTALL_PROBE = "!!(window.__ash && __ash.pump.install())"
PUMP_RESUME = "__ash.pump.resume()"


class _Conn:
    """Records CDP traffic; answers the pump-install probe."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.evals: list[str] = []
        self.closed = False

    def call(self, method, params=None, timeout=None):
        self.calls.append((method, params))
        return None

    def evaluate(self, expr, timeout=None):
        self.evals.append(expr)
        if "pump.install" in expr:
            return True
        return None

    def close(self):
        self.closed = True


def _env(drive: str) -> CdpSpeedrunEnv:
    env = CdpSpeedrunEnv(auto_connect=False, drive=drive)
    env.conn = _Conn()  # type: ignore[assignment]
    env.safety = lambda: dict(GAMEPLAY)  # type: ignore[method-assign]
    env.start_screencast = lambda **kw: False  # type: ignore[method-assign]
    return env


def test_unknown_drive_mode_is_rejected():
    with pytest.raises(ValueError, match="drive must be"):
        CdpSpeedrunEnv(auto_connect=False, drive="turbo")


def test_realtime_install_starts_the_engines_loop():
    """Realtime driving needs a running ticker, and must make it run.

    pump.uninstall(true) early-returns when no pump is installed - exactly the
    state after a previous run left the ticker stopped - so relying on it let a
    whole round run against a frozen game.
    """
    env = _env("realtime")
    env.install()
    assert any("pump.resume" in e for e in env.conn.evals), (
        "realtime install must start the engine's loop, not just release the pump"
    )


def test_frozen_game_is_refused_in_realtime_mode():
    """A stopped ticker makes a round meaningless; it must be refused, loudly."""
    env = _env("realtime")
    env.safety = lambda: dict(GAMEPLAY, tickerRunning=False)  # type: ignore[method-assign]
    reason = env.unsafe_reason()
    assert reason and "ticker is not running" in reason


def test_frozen_game_is_allowed_in_pump_mode():
    """The pump drives frames itself, so a stopped ticker is expected there."""
    env = _env("pump")
    env.safety = lambda: dict(GAMEPLAY, tickerRunning=False)  # type: ignore[method-assign]
    assert env.unsafe_reason() is None


def test_realtime_install_never_installs_the_pump():
    env = _env("realtime")
    info = env.install()

    assert info["pump_installed"] is False
    assert env._pump_installed is False
    # Match the PROBE, not the substring: agent.js itself is evaluated and
    # naturally contains the text "pump.install".
    assert not any(PUMP_INSTALL_PROBE in e for e in env.conn.evals), (
        "realtime mode must not hand-drive the ticker"
    )
    assert any(PUMP_RESUME in e for e in env.conn.evals), (
        "and it must start the engine loop (releasing any stale pump)"
    )


def test_pump_mode_still_installs_the_pump():
    """Deterministic replay keeps working; only self-play changes."""
    env = _env("pump")
    info = env.install()
    assert info["pump_installed"] is True
    assert any(PUMP_INSTALL_PROBE in e for e in env.conn.evals)


def test_realtime_step_holds_the_key_and_never_pumps(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("ash.env.cdp_backend.time.sleep", lambda s: slept.append(s))

    env = _env("realtime")
    env.step_frame(ActionSpace.minimal().masks[4], frames=15, hold=True)

    events = [e for e in env.conn.evals if "pump.pump" in e]
    assert events == [], "realtime mode must not hand-drive frames"
    downs = [c for c in env.conn.calls if c[0] == "Input.dispatchKeyEvent"
             and c[1].get("type") == "rawKeyDown"]
    ups = [c for c in env.conn.calls if c[0] == "Input.dispatchKeyEvent"
           and c[1].get("type") == "keyUp"]
    assert downs and ups, "the action must actually be pressed and released"
    assert slept, "the key must be held for real time, not released immediately"
    assert slept[0] == pytest.approx(env.config.frame_ms * 15 / 1000.0)


def test_realtime_hold_duration_matches_the_control_interval():
    """15 frames at 60 fps is the 0.25 s control interval, by construction."""
    from ash.config import load_game_config

    game = load_game_config()
    env = CdpSpeedrunEnv(auto_connect=False, drive="realtime")
    duration = env.config.frame_ms * game.control_frame_skip / 1000.0
    assert duration == pytest.approx(game.control_interval_s, abs=1e-6)


def test_realtime_close_still_pauses_by_stopping_the_ticker():
    """The paused exit is deliberate and must survive the drive-mode change."""
    env = _env("realtime")
    conn = env.conn
    env.close()  # resume defaults to False
    assert env.conn is None
    assert conn.closed is True
    assert any("ticker.stop()" in e for e in conn.evals), (
        "without a pump, pausing means stopping the engine's ticker"
    )


def test_realtime_close_with_resume_does_not_stop_the_ticker():
    env = _env("realtime")
    conn = env.conn
    env.close(resume=True)
    assert conn.closed is True
    assert not any("ticker.stop()" in e for e in conn.evals)


def test_runner_and_env_agree_on_the_timestep():
    """The runner's frame_skip must line up with the env's real-time step."""
    from ash.config import load_game_config
    from ash.loop.runner import InferenceRunner
    from ash.env.fake_backend import FakeSpeedrunEnv
    from ash.memory.kdm import KeyMomentModel
    from ash.models.ash_policy import AshPolicy, AshPolicyConfig

    class _Emb:
        def embed(self, frames, batch_size=64):
            import numpy as np
            return np.zeros((len(frames), 4), dtype=np.float32)

    env = FakeSpeedrunEnv()
    kdm = KeyMomentModel()
    kdm.embedder = _Emb()
    kdm.observe = lambda e, s: (False, -1, False)  # type: ignore[method-assign]
    policy = AshPolicy(AshPolicyConfig(image_size=32, w_s=4, w_l=2, num_layers=1))
    runner = InferenceRunner(
        lambda: env, policy, kdm, action_space=env.action_space,
        w_s=4, w_l=2, image_size=32, device="cpu",
        frame_skip=load_game_config().control_frame_skip,
    )
    assert runner.frame_skip == 15
