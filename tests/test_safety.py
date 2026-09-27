"""Input safety: never press a gameplay key outside Scene_Map.

RPG Maker's ok/cancel keys are the same physical keys the policy uses for
jump/attack.  On the title screen the highlighted entry is "continue", so a
single jump press loads the save; inside the pause menu the same press commits
a selection.  An unattended live run did exactly that and loaded the player's
save file.

These tests pin the three layers that now prevent it:

- the in-page probe reports the scene and fails closed when it cannot be read;
- the backend refuses to dispatch keys outside Scene_Map, whatever the caller
  intended;
- the runner ends the round instead of acting, and reports why.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ash.actions.space import ActionSpace  # noqa: E402
from ash.capture.screencast import CdpError, UnsafeSceneError  # noqa: E402
from ash.env.cdp_backend import CdpSpeedrunEnv  # noqa: E402

GAMEPLAY = {"scene": "Scene_Map", "messageBusy": False, "inGameplay": True}
TITLE = {"scene": "Scene_Title", "messageBusy": False, "inGameplay": False}
CONFIRM = {"scene": "Scene_Transport", "messageBusy": False,
           "inGameplay": False, "confirm": True}


class _FakeConn:
    """Records CDP calls; answers the safety expression with a canned payload."""

    def __init__(self, payload: dict | None) -> None:
        self.payload = None if payload is None else json.dumps(payload)
        self.calls: list[tuple[str, object]] = []
        self.evals: list[str] = []
        self.closed = False

    def call(self, method, params=None, timeout=None):
        self.calls.append((method, params))
        return None

    def evaluate(self, expr, timeout=None):
        self.evals.append(expr)
        if "__ash.safety" in expr:
            return self.payload
        if "pump.resume" in expr:
            return json.dumps({"resumed": True, "reason": None})
        return 0

    def close(self):
        self.closed = True


def _env(conn_payload: dict | None, *, enforce: bool = True) -> CdpSpeedrunEnv:
    """A CDP env that never connects; only the safety plumbing is exercised."""
    env = CdpSpeedrunEnv(auto_connect=False, enforce_safety=enforce)
    env.conn = _FakeConn(conn_payload)  # type: ignore[assignment]
    env._pump_installed = False
    return env


def test_safe_scene_reports_no_objection():
    assert _env(GAMEPLAY).unsafe_reason() is None


@pytest.mark.parametrize(
    "scene",
    ["Scene_Title", "Scene_Menu", "Scene_Load", "Scene_Save", "Scene_File",
     "Scene_Options", "Scene_Gameover", "Scene_Boot"],
)
def test_every_non_gameplay_scene_is_unsafe(scene):
    """The scenes that interpret a jump press as "confirm" must all be refused."""
    reason = _env({"scene": scene, "messageBusy": False, "inGameplay": False}).unsafe_reason()
    assert reason and scene in reason


def test_missing_agent_is_unsafe():
    """No probe answer is not permission - it is an unknown scene."""
    assert _env(None).unsafe_reason()


def test_unreadable_safety_fails_closed():
    """A probe that errors must block input, not wave it through."""
    env = CdpSpeedrunEnv(auto_connect=False)
    env.conn = None  # type: ignore[assignment]

    def boom():
        raise CdpError("websocket died")

    env.safety = boom  # type: ignore[method-assign]
    assert env.unsafe_reason()


def test_backend_refuses_to_dispatch_on_unsafe_scene():
    """The backstop: a caller that skips the runner's check still cannot press."""
    env = _env(TITLE)
    with pytest.raises(UnsafeSceneError, match="Scene_Title"):
        env.apply_action(0)
    assert env.conn.calls == [], "no key may be dispatched"


def test_backend_dispatches_on_gameplay_scene():
    env = _env(GAMEPLAY)
    env._check_safe()  # must not raise


def test_enforce_safety_false_is_an_explicit_opt_out():
    """Only the diagnostic probes opt out, and they must say so."""
    env = _env(TITLE, enforce=False)
    env._check_safe()  # must not raise


def test_runner_aborts_before_pressing_any_key():
    """A whole round must stop rather than act once on an unsafe screen."""
    hdbscan = pytest.importorskip("hdbscan")  # noqa: F841
    from ash.env.fake_backend import FakeSpeedrunEnv
    from ash.loop.runner import InferenceRunner
    from ash.memory.kdm import KeyMomentModel
    from ash.models.ash_policy import AshPolicy, AshPolicyConfig

    env = FakeSpeedrunEnv()
    stepped: list[object] = []
    env.step = lambda *a, **k: stepped.append(a)  # type: ignore[method-assign]
    env.unsafe_reason = lambda: "scene 'Scene_Title' is not Scene_Map gameplay"  # type: ignore[method-assign]

    policy = AshPolicy(AshPolicyConfig(image_size=32, w_s=4, w_l=2, num_layers=1))
    runner = InferenceRunner(
        lambda: env, policy, KeyMomentModel(),
        action_space=env.action_space, w_s=4, w_l=2, image_size=32, device="cpu",
    )
    out = runner.run([env], delta=6, timeout_s=5)

    assert out["aborted"] and "Scene_Title" in out["aborted"]
    assert stepped == [], "the runner must not step after a safety objection"
    assert out["steps"] == [0]


def test_runner_ignores_backends_without_the_hook():
    """The fake backend has no scenes, so it must be unaffected."""
    from ash.env.fake_backend import FakeSpeedrunEnv

    assert not hasattr(FakeSpeedrunEnv(), "unsafe_reason")


def test_close_leaves_the_game_paused_by_design():
    """Stopping the ticker on exit is deliberate, not a bug to "fix".

    An unattended character standing where the run ended is beaten to death by
    this engine's enemies, so the paused exit is the safe outcome.  It is easy
    to misread as a frozen game, so the intent is pinned here: the CLI must
    keep pausing (and say so in the log) rather than helpfully resuming.
    """
    env = CdpSpeedrunEnv(auto_connect=False)
    conn = _FakeConn(GAMEPLAY)
    env.conn = conn  # type: ignore[assignment]
    env._pump_installed = True

    env.close()  # resume defaults to False

    assert env.conn is None, "the socket must be released"
    assert conn.closed is True
    assert any("uninstall(false)" in e for e in conn.evals), (
        "close() must stop the ticker, leaving the character safe"
    )


def test_close_resume_true_is_the_explicit_handover():
    """A human taking the keyboard back must ask for it explicitly."""
    env = CdpSpeedrunEnv(auto_connect=False)
    conn = _FakeConn(GAMEPLAY)
    env.conn = conn  # type: ignore[assignment]
    env._pump_installed = True

    env.close(resume=True)

    assert any("uninstall(true)" in e for e in conn.evals)


def _confirm_env(presses_needed: int | None):
    """A fake env parked in an input-gated confirm scene; None = never clears."""
    from ash.env.fake_backend import FakeSpeedrunEnv

    env = FakeSpeedrunEnv()
    left = {"n": presses_needed or 0}
    env.unsafe_reason = lambda: (  # type: ignore[method-assign]
        "scene 'Scene_Transport' is not Scene_Map gameplay"
        if presses_needed is None or left["n"] > 0 else None
    )
    env.is_confirm_scene = lambda: presses_needed is None or left["n"] > 0  # type: ignore[method-assign]
    env.press_ok = lambda: (left.__setitem__("n", left["n"] - 1), True)[1]  # type: ignore[method-assign]
    return env


class _Emb:
    """Cheap stand-in for DINOv2: the runner only needs a vector per frame."""

    def embed(self, frames, batch_size=64):
        return np.zeros((len(frames), 4), dtype=np.float32)


def _runner_for(env, **kw):
    from ash.loop.runner import InferenceRunner
    from ash.memory.kdm import KeyMomentModel
    from ash.models.ash_policy import AshPolicy, AshPolicyConfig

    kdm = KeyMomentModel()
    kdm.embedder = _Emb()
    kdm.classify = lambda embedding, seen: False   # type: ignore[method-assign]
    kdm.cluster_of = lambda embedding: -1          # type: ignore[method-assign]

    policy = AshPolicy(AshPolicyConfig(image_size=32, w_s=4, w_l=2, num_layers=1))
    return InferenceRunner(
        lambda: env, policy, kdm,
        action_space=env.action_space, w_s=4, w_l=2, image_size=32, device="cpu", **kw,
    )


def test_runner_presses_ok_once_to_clear_a_confirm_scene():
    """An input-gated transition screen must be cleared, not avoided.

    Refusing all input on it does not protect anything: the screen never
    advances on its own, so a live game was parked there permanently.
    """
    pytest.importorskip("hdbscan")
    env = _confirm_env(presses_needed=1)

    out = _runner_for(env, frame_skip=1).run([env], delta=6, timeout_s=10)

    assert out["aborted"] is None, "one ok clears it; the round must continue"
    assert out["steps"][0] > 0


def test_runner_gives_up_when_the_confirm_scene_never_clears():
    """A screen that ignores three oks is not a transition screen."""
    pytest.importorskip("hdbscan")
    env = _confirm_env(presses_needed=None)

    out = _runner_for(env, frame_skip=1, max_confirm_presses=2).run(
        [env], delta=6, timeout_s=10
    )

    assert out["aborted"] and "did not clear after 2" in out["aborted"]
    assert out["steps"] == [0]


def test_press_ok_refuses_a_scene_outside_the_allow_list():
    """The exception must be exactly one allow-listed scene, not a loophole."""
    env = _env(TITLE)          # Scene_Title: ok there selects a menu entry
    with pytest.raises(UnsafeSceneError, match="allow-list"):
        env.press_ok()
    assert env.conn.calls == [], "no key may be dispatched on a menu"


def test_press_ok_dispatches_only_the_configured_ok_button():
    env = _env(GAMEPLAY)
    env.conn = _FakeConn(CONFIRM)  # type: ignore[assignment]
    env.press_ok()

    keys = [c for c in env.conn.calls if c[0] == "Input.dispatchKeyEvent"]
    assert keys, "the ok button must actually be pressed"
    pressed = {c[1]["key"] for c in keys if c[1].get("type") == "rawKeyDown"}
    assert pressed and pressed <= {"z"}, "only the ok key, never a random action"


def test_confirm_flag_comes_from_the_probe():
    assert _env(CONFIRM).is_confirm_scene() is True
    assert _env(TITLE).is_confirm_scene() is False
    assert _env(GAMEPLAY).is_confirm_scene() is False


def test_action_space_has_no_menu_escape_key():
    """Documented invariant: the live action space never contains ESC/menu.

    It cannot save us on its own - jump is Z, and Z is also "confirm" - which is
    exactly why the scene gate above exists.  But if ESC ever entered the space
    the agent could open the pause menu at will, so the invariant is pinned.
    """
    from ash.actions.space import buttons_from_mask

    space = ActionSpace.minimal()
    pressed = {b for mask in space.masks for b in buttons_from_mask(mask)}
    assert "menu" not in pressed
