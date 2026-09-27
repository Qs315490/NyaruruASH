"""GAME OVER: moving the cursor onto the save load, and never onto the other one.

The screen is a two-button box.  `up` is STATIC_TEXT_CONTINUE_YES (load the save,
the operator-verified safe action) and `down` is STATIC_TEXT_CONTINUE_NO (back to
town).  The guardrail refuses the whole screen unless the cursor is already on the
save load, which is correct - but it also meant every death threw the round away:
measured, a 2500-step collection ended after 1314 steps with

    a menu entry that must not be committed is highlighted:
    [{'panel': '_selectBox', 'name': 'STATIC_TEXT_CONTINUE_NO', 'index': 'down'}]

There WAS an automatic routine for this, and it had two problems: nothing called
it, and it selected `down` - the entry the policy refuses.  Both are fixed here,
and these tests pin the parts that matter:

  * the first key dispatched moves the cursor onto the save-load entry, so no ok
    can ever land on the refused one;
  * the outcome is verified by re-reading the scene, never assumed;
  * anything unreadable or off-screen is refused without pressing anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from ash.actions.space import ActionSpace
from ash.env.cdp_backend import CdpSpeedrunEnv

GAMEPLAY = {"scene": "Scene_Map", "messageBusy": False, "inGameplay": True,
            "awaitingChoice": False, "confirm": False, "tickerRunning": True,
            "menuEntries": [], "guardedMenuEntry": False, "menuOperable": False}


def _gameover(entry: str | None, index: str = "down") -> dict:
    entries = [] if entry is None else [
        {"panel": "_selectBox", "index": index, "name": entry}]
    return dict(GAMEPLAY, scene="Scene_Gameover", inGameplay=False,
                menuOperable=True, menuEntries=entries,
                guardedMenuEntry=entry == "STATIC_TEXT_CONTINUE_NO")


class _Conn:
    def __init__(self) -> None:
        self.keys: list[tuple[str, str]] = []

    def call(self, method, params=None, timeout=None):
        if method == "Input.dispatchKeyEvent" and isinstance(params, dict):
            self.keys.append((params.get("key"), params.get("type")))
        return None

    def evaluate(self, expr, timeout=None):
        return None

    def close(self):
        return None


class _Scripted:
    """Returns the next safety payload each read, sticking on the last one.

    The recovery has to observe the world change, so a fake that always answers
    the same thing can never let it succeed - and that is exactly how the
    fail-closed test is written.
    """

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls = 0

    def __call__(self) -> dict:
        self.calls += 1
        return self.payloads[min(self.calls - 1, len(self.payloads) - 1)]


def _env(payloads: list[dict]) -> CdpSpeedrunEnv:
    env = CdpSpeedrunEnv(auto_connect=False, drive="realtime")
    env.conn = _Conn()                     # type: ignore[assignment]
    env.config.frame_ms = 0.0
    env.safety = _Scripted(payloads)       # type: ignore[method-assign]
    env.unsafe_reason = lambda: None       # type: ignore[method-assign]
    env._frame = np.zeros((8, 8, 3), dtype=np.uint8)
    return env


def env_max_presses() -> int:
    """The bound the recovery promises; asserted against rather than a literal."""
    import inspect

    return inspect.signature(CdpSpeedrunEnv.resolve_gameover).parameters["max_presses"].default


def _up_key(env: CdpSpeedrunEnv) -> str:
    from ash.actions.space import mask_from_buttons

    return env._bindings_for(mask_from_buttons(["up"]))[0].single()[0]


def _ok_key(env: CdpSpeedrunEnv) -> str:
    return env._bindings_for(env._confirm_mask())[0].single()[0]


def test_pending_is_true_only_when_the_cursor_sits_on_the_refused_entry():
    assert _env([_gameover("STATIC_TEXT_CONTINUE_NO")]).gameover_recovery_pending()
    assert not _env([_gameover("STATIC_TEXT_CONTINUE_YES", "up")]).gameover_recovery_pending()
    assert not _env([GAMEPLAY]).gameover_recovery_pending()
    assert not _env([_gameover(None)]).gameover_recovery_pending()


def test_the_cursor_moves_onto_the_save_load_before_any_ok():
    """The first key dispatched must be `up`.  Order is the whole safety claim."""
    unsafe = _gameover("STATIC_TEXT_CONTINUE_NO", "down")
    safe = _gameover("STATIC_TEXT_CONTINUE_YES", "up")
    # reads: identify, then the poll after `up`, then the poll after `ok`
    env = _env([unsafe, unsafe, safe, safe, dict(GAMEPLAY)])
    out = env.resolve_gameover()

    downs = [k for k, t in env.conn.keys if t == "rawKeyDown"]
    assert downs, "the recovery dispatched nothing"
    assert downs[0] == _up_key(env), "the first press must move onto the save load"
    assert _ok_key(env) not in downs[:1], "ok must never land on the refused entry"
    assert out["resolved"] is True, out
    assert out["presses"] >= 2


def test_success_is_verified_not_assumed():
    """A screen that never changes must NOT be reported as recovered.

    This is the failure the fail-closed rule exists for: claiming success and
    then dispatching gameplay keys onto a screen where ok commits a choice.
    The implementation refuses even earlier than that - if the cursor did not
    move, it never presses ok at all - so either refusal is acceptable and the
    assertion is on what must NOT happen: success, or an unbounded sequence.
    """
    env = _env([_gameover("STATIC_TEXT_CONTINUE_NO")])   # stuck forever
    out = env.resolve_gameover()
    assert out["resolved"] is False, out
    assert out["reason"], "a refusal must say why"
    assert out["presses"] <= env_max_presses(), "presses must stay bounded"
    ok = _ok_key(env)
    assert ok not in [k for k, t in env.conn.keys if t == "rawKeyDown"], (
        "ok must not be pressed on a screen whose cursor never moved")


def test_an_unreadable_cursor_presses_nothing():
    env = _env([_gameover(None)])
    out = env.resolve_gameover()
    assert out["resolved"] is False
    assert "could not be read" in (out["reason"] or ""), out
    assert env.conn.keys == [], "an unreadable cursor must not be guessed at"


def test_it_refuses_anywhere_except_the_game_over_screen():
    env = _env([GAMEPLAY])
    out = env.resolve_gameover()
    assert out["resolved"] is False and "not on the game over screen" in out["reason"]
    assert env.conn.keys == []


def test_already_on_the_safe_entry_needs_no_recovery():
    env = _env([_gameover("STATIC_TEXT_CONTINUE_YES", "up")])
    out = env.resolve_gameover()
    assert out["resolved"] is False
    assert "already highlighted" in out["reason"], out
    assert env.conn.keys == []


def test_the_narrow_gate_allows_the_game_over_screen_only():
    """`gameover_reason` is the narrow mode's verdict, and it is not a bypass."""
    on_screen = _env([_gameover("STATIC_TEXT_CONTINUE_NO")])
    assert on_screen.gameover_reason() is None
    elsewhere = _env([GAMEPLAY])
    assert "not on the game over screen" in (elsewhere.gameover_reason() or "")
    unknown = _env([_gameover("STATIC_TEXT_SOMETHING_ELSE")])
    assert "unrecognised entry" in (unknown.gameover_reason() or "")


def test_the_runner_gate_routes_a_death_to_the_recovery_not_an_abort():
    from ash.loop.runner import InferenceRunner

    env = _env([_gameover("STATIC_TEXT_CONTINUE_NO")])
    env.unsafe_reason = lambda: ("a menu entry that must not be committed is "
                                 "highlighted: [...]")   # type: ignore[method-assign]
    kind, reason = InferenceRunner._scene_gate([env])
    assert kind == "gameover", (kind, reason)
