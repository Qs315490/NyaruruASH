"""A menu may be left with cancel - and only with cancel.

The agent presses its way into menus by accident (measured: Scene_SkillSt after
39 steps of a real round).  Gameplay input is refused outside Scene_Map, and it
must stay refused - ok IS the jump key and commits selections - but an agent
that can never leave a menu loses the whole round and needs a human every time.
The escape is therefore a separate, narrower capability: a scene agent.js lists
as a menu, the cancel button, no directions, no ok, and a cap.

The list is NOT a confirm list.  ok committing a selection is still the thing
that got an unattended run into the player's save, and no scene may be on both.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ash.env.fake_backend import FakeSpeedrunEnv

AGENT_JS = Path(__file__).resolve().parent.parent / "src" / "ash" / "memory" / "agent.js"


def _lists() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    script = (
        "global.window = global;"
        "global.document = {addEventListener: function () {}};"
        "global.navigator = {};"
        "eval(require('fs').readFileSync(%r, 'utf8'));"
        "console.log(JSON.stringify({menu: window.__ash.MENU_SCENES,"
        " confirm: window.__ash.CONFIRM_SCENES}));" % str(AGENT_JS)
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-600:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_menu_and_confirm_lists_are_disjoint():
    """ok-safe and cancel-safe are different questions and must not be merged."""
    lists = _lists()
    # An entry now exists - Scene_ItemObtain, whose ok the game binds to popScene -
    # but the two lists answer different questions and still must not overlap.
    assert lists["confirm"] == ["Scene_ItemObtain"], lists["confirm"]
    assert not set(lists["menu"]) & set(lists["confirm"])
    assert "Scene_SkillSt" in lists["menu"], "this is the scene that trapped a real round"


class _MenuEnv(FakeSpeedrunEnv):
    """The real fake backend, reporting a menu scene the runner must escape."""

    def __init__(self) -> None:
        super().__init__()
        self.presses = 0
        self.in_menu = True

    def unsafe_reason(self):
        if self.in_menu:
            return "scene 'Scene_SkillSt' is not Scene_Map gameplay"
        return None

    def safety(self):
        if self.in_menu:
            return {"scene": "Scene_SkillSt", "menu": True, "inGameplay": False,
                    "awaitingChoice": False, "messageBusy": False, "tickerRunning": True}
        return {"scene": "Scene_Map", "menu": False, "inGameplay": True,
                "awaitingChoice": False, "messageBusy": False, "tickerRunning": True}

    def escape_menu(self, *, max_presses=4):
        self.presses += 1
        self.in_menu = False
        return {"escaped": True, "presses": 1, "scene": "Scene_Map", "reason": None}


class _StuckMenuEnv(_MenuEnv):
    """A menu the escape never clears: the round must end rather than spin."""

    def escape_menu(self, *, max_presses=4):
        self.presses += 1
        return {"escaped": False, "presses": max_presses,
                "scene": "Scene_SkillSt", "reason": "still in the menu"}


def test_escape_refuses_while_a_choice_is_pending():
    from ash.env.cdp_backend import CdpSpeedrunEnv

    env = CdpSpeedrunEnv(auto_connect=False)
    env.enforce_safety = True
    env._state_override = None
    env.safety = lambda: {"scene": "Scene_SkillSt", "menu": True, "inGameplay": False,
                          "awaitingChoice": True, "messageBusy": False,
                          "tickerRunning": True}  # type: ignore[method-assign]
    reason = env.menu_escape_reason()
    assert reason and "choice" in reason


def test_escape_refuses_an_unknown_scene():
    from ash.env.cdp_backend import CdpSpeedrunEnv

    env = CdpSpeedrunEnv(auto_connect=False)
    env.safety = lambda: {"scene": "Scene_SomethingCustom", "menu": False,
                          "inGameplay": False, "awaitingChoice": False,
                          "messageBusy": False, "tickerRunning": True}  # type: ignore[method-assign]
    reason = env.menu_escape_reason()
    assert reason and "not a known menu" in reason


def test_escape_presses_cancel_only_and_stops_at_gameplay():
    from ash.actions.space import buttons_from_mask
    from ash.env.cdp_backend import CdpSpeedrunEnv

    env = CdpSpeedrunEnv(auto_connect=False)
    env.safety = lambda: {"scene": "Scene_SkillSt", "menu": True, "inGameplay": False,
                          "awaitingChoice": False, "messageBusy": False,
                          "tickerRunning": True}  # type: ignore[method-assign]
    seen = []
    env._dispatch = lambda mask: seen.append(buttons_from_mask(mask))  # type: ignore[method-assign]

    def safety_after():
        # Pressing cancel is what returns to gameplay, so the scene flips only
        # once a press has actually been dispatched.
        if seen:
            return {"scene": "Scene_Map", "menu": False, "inGameplay": True,
                    "awaitingChoice": False, "messageBusy": False, "tickerRunning": True}
        return {"scene": "Scene_SkillSt", "menu": True, "inGameplay": False,
                "awaitingChoice": False, "messageBusy": False, "tickerRunning": True}

    env.safety = safety_after  # type: ignore[method-assign]
    out = env.escape_menu(max_presses=5)
    assert out["escaped"] is True
    assert out["presses"] == 1, "must stop the moment gameplay is reached"
    assert seen == [("cancel",)], "cancel is the only button the escape may press"


class _KdmStub:
    class _Emb:
        def embed(self, frames, batch_size=64):
            import numpy as np
            return np.zeros((len(frames), 4), dtype="float32")

    embedder = _Emb()

    def observe(self, embedding, seen):
        return False, -1, False

    def classify_sequence(self, embeddings):
        import numpy as np
        return np.zeros(len(embeddings), dtype=bool)


def _menu_runner(env, escapes=3):
    from ash.loop.runner import InferenceRunner
    from ash.models.ash_policy import AshPolicy, AshPolicyConfig

    space = env.action_space
    policy = AshPolicy(AshPolicyConfig(image_size=16, w_s=4, w_l=2, num_layers=1,
                                       num_actions=len(space)))
    return InferenceRunner(
        lambda: env, policy, _KdmStub(), action_space=space,
        w_s=4, w_l=2, image_size=16, device="cpu", max_menu_escapes=escapes,
    )


def test_runner_escapes_a_menu_instead_of_losing_the_round():
    """A menu must cost a few cancel presses, not the whole round."""
    env = _MenuEnv()
    out = _menu_runner(env).run([env], delta=4, timeout_s=20)
    assert env.presses >= 1, "the runner must have tried the cancel escape"
    assert out["aborted"] is None, "a recoverable menu must not abort the round"


def test_runner_gives_up_after_the_escape_cap():
    """A scene that keeps returning is a loop; the round must end, not spin."""
    env = _StuckMenuEnv()
    out = _menu_runner(env, escapes=3).run([env], delta=4, timeout_s=20)
    assert env.presses <= 4, "the escape must be bounded, not retried forever"
    assert out["aborted"], "an unescapable menu must still abort"
