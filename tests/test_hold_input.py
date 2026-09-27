"""Holding a key ACROSS steps is what makes this game's jump expressible.

Measured with the engine's own py: jump height is set by how long the button
stays down, and one control interval (250 ms) reaches 172 px where the game
allows 212.  The second jump needs a release at the apex before the next press.
A step that unconditionally releases - the old contract, and the right default -
can express neither, so every jump the agent has taken has been a short one and
the double jump has been impossible rather than unlearned.

These tests pin the held-input contract and, just as importantly, the release
guarantees that make holding safe.  The old code's comment is explicit about why
the release was unconditional: a game that never sees a keyup keeps the button
down, so a single mistake becomes a stuck character.  Holding makes that contract
conditional, and a conditional release is only as good as its worst exit path.
"""

from __future__ import annotations

import pytest

import numpy as np

from ash.actions.space import ActionSpace, buttons_from_mask
from ash.env.cdp_backend import CdpSpeedrunEnv, UnsafeSceneError

JUMP = next(i for i, m in enumerate(ActionSpace.minimal().masks)
            if buttons_from_mask(m) == ("jump",))

GAMEPLAY = {"scene": "Scene_Map", "messageBusy": False, "inGameplay": True,
            "awaitingChoice": False, "confirm": False, "tickerRunning": True}


class _Conn:
    """Records every dispatched key event, in order."""

    def __init__(self) -> None:
        self.keys: list[tuple[str, str]] = []      # (key name, type)

    def call(self, method, params=None, timeout=None):
        if method == "Input.dispatchKeyEvent" and isinstance(params, dict):
            self.keys.append((params.get("key"), params.get("type")))
        return None

    def evaluate(self, expr, timeout=None):
        return None

    def close(self):
        return None


def _env(*, unsafe: str | None = None) -> CdpSpeedrunEnv:
    env = CdpSpeedrunEnv(auto_connect=False, drive="realtime")
    env.conn = _Conn()  # type: ignore[assignment]
    # `close()` detaches the connection, so the recorder is kept separately and
    # the assertions read it rather than the env's now-None attribute.
    env._recorder = env.conn
    env.safety = lambda: dict(GAMEPLAY)  # type: ignore[method-assign]
    env.config.frame_ms = 0.0          # no sleeping in tests
    # `step()` observes, which takes a real screenshot; the key events are what
    # these tests are about, so the picture is a stub.
    def _shot(**kw):
        env._frame = np.zeros((8, 8, 3), dtype=np.uint8)
        return env._frame

    env.capture_once = _shot                                           # type: ignore
    env.milestone = lambda: None       # type: ignore[method-assign]
    env.game_frame_count = lambda: 0   # type: ignore[method-assign]
    if unsafe is not None:
        env.unsafe_reason = lambda: unsafe  # type: ignore[method-assign]
    return env


def _downs(env) -> list[str]:
    return [k for k, t in env._recorder.keys if t == "rawKeyDown"]


def _ups(env) -> list[str]:
    return [k for k, t in env._recorder.keys if t == "keyUp"]


def test_repeating_an_action_keeps_the_key_down():
    """This is the mechanism: same action again = a longer hold, not a new tap."""
    env = _env()
    space = env.action_space
    jump = JUMP
    env.step_holding(env.action_space.mask_at(jump), frames=15)
    assert len(_downs(env)) == 1 and _ups(env) == [], "first step presses and holds"
    env.step_holding(env.action_space.mask_at(jump), frames=15)
    assert len(_downs(env)) == 1, "the same action again must NOT re-press"
    assert _ups(env) == [], "and must NOT release: this is the longer jump"
    env.step_holding(env.action_space.mask_at(jump), frames=15)
    assert len(_ups(env)) == 0
    assert env.held_action == space.mask_at(jump)


def test_switching_action_releases_then_presses():
    """`jump -> noop -> jump` has to produce a real release and re-press."""
    env = _env()
    space = env.action_space
    jump = JUMP
    env.step_holding(env.action_space.mask_at(jump), frames=15)
    env.step_holding(space.mask_at(0), frames=15)                 # noop: nothing held, but...
    assert _ups(env), "switching away must release the held button"
    env.step_holding(env.action_space.mask_at(jump), frames=15)
    assert len(_downs(env)) == 2, "the second press is a NEW keydown: the double jump"


def test_release_is_idempotent_and_never_gated():
    """Letting go must always be allowed, including inside a menu.

    `press()` refuses outside Scene_Map because the same keys confirm menus.
    `release()` cannot share that gate: refusing to lift a button inside a menu
    would leave it stuck down for the rest of the session.
    """
    env = _env()
    jump = JUMP
    env.step_holding(env.action_space.mask_at(jump), frames=15)
    assert _ups(env) == []
    env.release()
    assert len(_ups(env)) == 1 and env.held_action is None
    env.release()
    assert len(_ups(env)) == 1, "a second release dispatches nothing"

    env2 = _env(unsafe="scene 'Scene_Menu' is not Scene_Map gameplay")
    env2._held_action = 0                 # pretend a button was already down
    env2._held_bindings = []
    env2.release()                        # must not raise
    assert env2.held_action is None


def test_press_is_refused_on_an_unsafe_scene_and_dispatch_is_empty():
    env = _env(unsafe="scene 'Scene_Menu' is not Scene_Map gameplay")
    with pytest.raises(UnsafeSceneError):
        env.press(1)
    assert env._recorder.keys == [], "a refused press must not touch the keyboard"


def test_non_holding_dispatch_ends_any_hold():
    """`press_ok` and `escape_menu` go through step_frame; they must not leave a
    movement key down while they press confirm or cancel."""
    env = _env()
    jump = JUMP
    env.step_holding(env.action_space.mask_at(jump), frames=15)
    before = len(_ups(env))
    env.step_frame(env.action_space.noop, frames=15)     # a plain, releasing step
    assert len(_ups(env)) == before + 1, "a non-holding dispatch must lift the hold"
    assert env.held_action is None


def test_close_releases_before_pausing():
    """Leaving a movement key down while paused means the character walks off a
    ledge the moment a human resumes the ticker."""
    env = _env()
    jump = JUMP
    env.step_holding(env.action_space.mask_at(jump), frames=15)
    env.close(resume=False)
    assert env.held_action is None and _ups(env), "close() must lift what it held"


def test_default_path_still_releases_every_step():
    """The old behaviour is the default: no hold, so nothing may stay down."""
    env = _env()
    space = env.action_space
    for i in (1, 2, 1):
        env.step(space.mask_at(i), frames=15)
    assert env.held_action is None
    assert len(_downs(env)) == 3 and len(_ups(env)) == 3, "one press and one release each"
