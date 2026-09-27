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

    def __init__(self, payload) -> None:
        # A list is served in order, so a test can model the state *after* a
        # press (the difficulty resolver verifies instead of assuming).  A single
        # payload stays constant, which is what every other test wants.
        if isinstance(payload, list):
            self.payloads = [None if p is None else json.dumps(p) for p in payload]
        else:
            self.payloads = None if payload is None else [json.dumps(payload)]
        self.calls: list[tuple[str, object]] = []
        self.evals: list[str] = []
        self.events: list[str] = []   # ordered, for ordering assertions
        self.closed = False

    def call(self, method, params=None, timeout=None):
        self.calls.append((method, params))
        self.events.append("call:%s:%s" % (method, (params or {}).get("type", "")))
        return None

    def evaluate(self, expr, timeout=None):
        self.evals.append(expr)
        self.events.append("eval:%s" % expr)
        if "__ash.safety" in expr:
            if not self.payloads:
                return None
            return self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
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
    kdm.observe = lambda embedding, seen: (False, -1, False)  # type: ignore[method-assign]

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


def test_press_ok_holds_the_key_across_a_frame():
    """Down and up must straddle a game frame, or the confirm edge is missed.

    The engine samples Input on the frames it runs; dispatching both events
    back-to-back leaves every frame seeing an idle keyboard.
    """
    env = _env(GAMEPLAY)
    env.conn = _FakeConn(CONFIRM)  # type: ignore[assignment]
    env._pump_installed = True
    env.press_ok()

    down = next(i for i, e in enumerate(env.conn.events) if "rawKeyDown" in e)
    up = next(i for i, e in enumerate(env.conn.events) if "keyUp" in e)
    pumps = [i for i, e in enumerate(env.conn.events) if "pump.pump" in e]
    assert any(down < p < up for p in pumps), "the key must be held across a frame"
    assert any(p > up for p in pumps), "and released on a later frame"


def test_scene_map_with_a_pending_choice_is_unsafe():
    """Scene_Map is not enough: ok answers a waiting choice.

    The payload carries no `choices`, which is the unreadable case: the guard
    cannot tell an ordinary line from the difficulty pick, so it refuses.  The
    readable cases are pinned separately - an ordinary choice is allowed and the
    difficulty pick is not.
    """
    payload = {"scene": "Scene_Map", "messageBusy": True, "inGameplay": True,
               "awaitingChoice": True}
    env = _env(payload)

    reason = env.unsafe_reason()
    assert reason and "choice" in reason
    assert env.is_confirm_scene() is False, "a choice must not be ok-pressed away"
    with pytest.raises(UnsafeSceneError):
        env.press_ok()


def test_plain_dialogue_on_the_map_still_behaves_as_gameplay():
    """A message without choices is just text; it must not block the round."""
    payload = {"scene": "Scene_Map", "messageBusy": True, "inGameplay": True,
               "awaitingChoice": False}
    assert _env(payload).unsafe_reason() is None


def test_runner_aborts_on_a_pending_dialogue_choice():
    pytest.importorskip("hdbscan")
    from ash.env.fake_backend import FakeSpeedrunEnv

    env = FakeSpeedrunEnv()
    stepped: list[object] = []
    env.step = lambda *a, **k: stepped.append(a)  # type: ignore[method-assign]
    env.unsafe_reason = lambda: "a dialogue choice is awaiting an answer"  # type: ignore[method-assign]

    out = _runner_for(env, frame_skip=1).run([env], delta=6, timeout_s=5)

    assert out["aborted"] and "choice" in out["aborted"]
    assert stepped == [], "no key may be pressed while a choice is waiting"


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


def test_an_ordinary_dialogue_choice_is_the_agents_to_answer():
    """Refusing every choice made the story unplayable.

    The ok key is the policy's jump key, and choices are how the game advances,
    so the user's rule is that ordinary dialogue is the agent's to operate and
    the difficulty pick is not.
    """
    payload = {"scene": "Scene_Map", "messageBusy": True, "inGameplay": True,
               "awaitingChoice": True, "guardedChoice": False,
               "choices": ["是", "否"]}
    assert _env(payload).unsafe_reason() is None, "an ordinary choice must not block the round"


def test_the_difficulty_pick_is_still_refused():
    payload = {"scene": "Scene_Map", "messageBusy": True, "inGameplay": True,
               "awaitingChoice": True, "guardedChoice": True,
               "choices": ["按下简单难度按钮", "按下普通难度按钮", "\\C[18]按下困难难度按钮"]}
    reason = _env(payload).unsafe_reason()
    assert reason and "must not be answered" in reason
    with pytest.raises(UnsafeSceneError):
        env_press = _env(payload)
        env_press.press_ok()


def test_a_choice_whose_options_cannot_be_read_still_blocks():
    """Fail closed: an unclassifiable choice may be the one we must not answer."""
    payload = {"scene": "Scene_Map", "messageBusy": True, "inGameplay": True,
               "awaitingChoice": True, "guardedChoice": False, "choices": []}
    reason = _env(payload).unsafe_reason()
    assert reason and "could not be read" in reason


def test_guarded_choice_rule_in_the_shipped_agent_js():
    """The rule lives in JS, so it is loaded and executed, not re-implemented.

    The captured option texts are the evidence:
      ['按下简单难度按钮', '按下普通难度按钮', '\\C[18]按下困难难度按钮']
    The third carries an MZ colour code, which is why the matcher strips control
    codes first - without that the hard-difficulty option slips through.
    """
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    agent_js = __import__("pathlib").Path(__file__).resolve().parent.parent \
        / "src" / "ash" / "memory" / "agent.js"
    script = (
        "global.window = global;"
        "global.document = {addEventListener: function () {}};"
        "global.navigator = {};"
        "eval(require('fs').readFileSync(%r, 'utf8'));"
        "var V = window.__ash;"
        "console.log(JSON.stringify({"
        "  difficulty: V.guardedChoice(['按下简单难度按钮','按下普通难度按钮','\\\\C[18]按下困难难度按钮']),"
        "  coloured_only: V.guardedChoice(['\\\\C[18]按下困难难度按钮']),"
        "  ordinary: V.guardedChoice(['是','否']),"
        "  continuation: V.guardedChoice(['继续','跳过']),"
        "  stripped: V.stripControlCodes('\\\\C[18]按下困难难度按钮'),"
        "  guarded: V.GUARDED_CHOICES"
        "}));" % str(agent_js)
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, "agent.js failed in node:\n%s" % out.stderr[-600:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["difficulty"] is True, got
    assert got["coloured_only"] is True, "the colour code hid the hard option"
    assert got["ordinary"] is False, got
    assert got["continuation"] is False, got
    assert got["stripped"] == "按下困难难度按钮", got
    assert "难度" in got["guarded"], got


DIFFICULTY = {
    "scene": "Scene_Map", "messageBusy": True, "inGameplay": True,
    "tickerRunning": True, "awaitingChoice": True, "guardedChoice": True,
    "choices": ["按下简单难度按钮", "按下普通难度按钮", "\\C[18]按下困难难度按钮"],
    "choiceTexts": ["按下简单难度按钮", "按下普通难度按钮", "按下困难难度按钮"],
    "choiceIndex": 0,
}
NO_CHOICE = {"scene": "Scene_Map", "messageBusy": False, "inGameplay": True,
             "tickerRunning": True, "awaitingChoice": False, "guardedChoice": False,
             "choices": [], "choiceTexts": [], "choiceIndex": None}


class _DifficultyConn(_FakeConn):
    """A dialogue that stays waiting until the ok key is actually pressed.

    The resolver re-reads safety before every press and again afterwards, so a
    payload list would have to know that count.  Modelling the transition
    behaviourally tests the property that matters instead: the choice is gone
    because it was answered, not because reads ran out.
    """

    def __init__(self, waiting: dict, answered: dict) -> None:
        super().__init__(waiting)
        self._waiting = json.dumps(waiting)
        self._answered_state = json.dumps(answered)
        self.answered = False

    def call(self, method, params=None, timeout=None):
        out = super().call(method, params, timeout)
        if (method == "Input.dispatchKeyEvent"
                and (params or {}).get("type") == "rawKeyDown"
                and params.get("key") not in ("ArrowDown", "ArrowUp")):
            self.answered = True          # the ok key
        return out

    def evaluate(self, expr, timeout=None):
        if "__ash.safety" in expr:
            return self._answered_state if self.answered else self._waiting
        return super().evaluate(expr, timeout)


def _difficulty_env(waiting: dict, answered: dict | None = None):
    env = CdpSpeedrunEnv(auto_connect=False, enforce_safety=True)
    env.conn = _DifficultyConn(waiting, answered or NO_CHOICE)
    env._pump_installed = False
    return env


def _pressed(env) -> list:
    """The keys actually dispatched, in order, one entry per press."""
    return [p.get("key") for m, p in env.conn.calls
            if m == "Input.dispatchKeyEvent" and p.get("type") == "rawKeyDown"]


def test_without_a_preset_the_pick_is_still_refused():
    """Answering the difficulty pick is the operator's call, never a default."""
    env = _env(DIFFICULTY)
    assert env.difficulty_option is None
    assert "no difficulty preset" in env.difficulty_reason()
    out = env.resolve_difficulty()
    assert not out["resolved"] and out["reason"]
    assert _pressed(env) == [], "nothing may be dispatched without a preset"


def test_the_preset_difficulty_is_moved_to_and_confirmed():
    """0 -> 1 needs one 'down', then the ok key, then a verified read."""
    env = _difficulty_env(DIFFICULTY)
    env.difficulty_option = "普通"
    out = env.resolve_difficulty()
    assert out["resolved"] and out["option"] == 1 and out["presses"] == 2, out
    keys = _pressed(env)
    assert keys[0] == "ArrowDown", keys
    assert keys[1] not in ("ArrowDown", "ArrowUp"), "the second press is the ok key"


def test_the_preset_can_be_a_number_or_a_sanitised_text():
    """The hard option carries a colour code, so matching must use the clean text."""
    env = _difficulty_env(dict(DIFFICULTY, choiceIndex=0))
    env.difficulty_option = "困难"
    assert env.resolve_difficulty()["option"] == 2, "the colour code hid the option"

    env = _difficulty_env(dict(DIFFICULTY, choiceIndex=2))
    env.difficulty_option = 1              # 1-based: the first option, as typed
    out = env.resolve_difficulty()
    assert out["option"] == 0 and _pressed(env)[0] == "ArrowUp", _pressed(env)


def test_a_choice_that_is_not_the_difficulty_pick_cannot_be_answered():
    """This must never become a way to answer arbitrary dialogue."""
    payload = dict(DIFFICULTY, guardedChoice=False, choices=["是", "否"],
                   choiceTexts=["是", "否"])
    env = _env(payload)
    env.difficulty_option = "是"
    assert "not the guarded" in env.difficulty_reason()
    assert not env.resolve_difficulty()["resolved"]
    assert _pressed(env) == []


def test_a_preset_that_matches_nothing_dispatches_nothing():
    env = _env(DIFFICULTY)
    env.difficulty_option = "恶梦"
    out = env.resolve_difficulty()
    assert not out["resolved"] and "matches none" in out["reason"]
    assert _pressed(env) == []


def test_a_pick_that_survives_the_presses_is_not_reported_as_resolved():
    """Success is verified, not assumed - the choice must be gone afterwards."""
    env = _difficulty_env(DIFFICULTY, DIFFICULTY)   # still waiting after the ok
    env.difficulty_option = "普通"
    out = env.resolve_difficulty()
    assert not out["resolved"] and "still waiting" in out["reason"]


def test_an_unreadable_highlighted_index_is_refused():
    env = _difficulty_env(dict(DIFFICULTY, choiceIndex=None))
    env.difficulty_option = "普通"
    out = env.resolve_difficulty()
    assert not out["resolved"] and "could not be read" in out["reason"]


def _difficulty_runner_env(resolved: bool):
    """Stateful: a pick that is gone once it has been answered.

    A mock that reports "pending" forever would trip the resolve bound instead of
    testing the round, which is a different (and correct) behaviour.
    """
    from ash.env.fake_backend import FakeSpeedrunEnv

    env = FakeSpeedrunEnv()
    state = {"pending": True}
    env.unsafe_reason = lambda: (  # type: ignore[method-assign]
        "a dialogue choice that must not be answered" if state["pending"] else None)
    env.difficulty_choice_pending = lambda: state["pending"]  # type: ignore[method-assign]

    def resolve():
        if resolved:
            state["pending"] = False
        return {"resolved": resolved, "option": 1, "presses": 2,
                "reason": None if resolved else "still waiting"}

    env.resolve_difficulty = resolve  # type: ignore[method-assign]
    return env


def test_runner_answers_the_difficulty_pick_instead_of_aborting():
    """The dialogue cannot be closed any other way, so refusing parked the game.

    With a preset configured the round answers it and carries on; the choice is
    made deliberately by the operator, not by the policy.
    """
    env = _difficulty_runner_env(True)
    stepped: list = []
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    env.reset = lambda *a, **k: frame  # type: ignore[method-assign]
    env.step = lambda *a, **k: (stepped.append(a), frame)[1]  # type: ignore[method-assign]

    out = _runner_for(env, frame_skip=1).run([env], delta=4, timeout_s=5)

    assert not out["aborted"], out
    assert stepped, "the round must continue once the pick is answered"


def test_runner_aborts_when_the_pick_cannot_be_answered():
    """A failed resolve must abort, never fall through into the dialogue."""
    env = _difficulty_runner_env(False)
    stepped: list = []
    env.step = lambda *a, **k: stepped.append(a)  # type: ignore[method-assign]

    out = _runner_for(env, frame_skip=1).run([env], delta=4, timeout_s=5)

    assert out["aborted"] and "choice" in out["aborted"]
    assert stepped == [], "no gameplay key may be pressed on an open dialogue"


def test_the_difficulty_preset_comes_from_the_game_config():
    """Easy by default, because the corpus is speedruns and they are all easy.

    Training on easy runs while the agent plays another difficulty compares two
    different games; per-difficulty training means changing this file, not mixing
    the two.
    """
    from ash.config import load_game_config

    assert load_game_config().difficulty_preset == "简单"


def test_the_random_phase_reports_its_own_maps():
    """`maps` alone hid the run's actual progress.

    The first multi-round run reported `maps [[4]]` for all three rounds while the
    character had walked out to map 5 during a random rollout - the metric being
    read to decide "did it get out?" could not see the phase in which it did.
    The two phases are now reported separately, because "the policy walked there"
    and "random exploration happened to walk there" are different results.
    """
    from ash.env.fake_backend import FakeSpeedrunEnv

    env = FakeSpeedrunEnv()
    reads = {"n": 0}

    def state():
        reads["n"] += 1
        # Sampled every 10 steps, so a 40-step rollout reads this four times.
        return {"player": {"mapId": 4 if reads["n"] < 3 else 5}}

    env.state = state  # type: ignore[method-assign]
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    env.reset = lambda *a, **k: frame  # type: ignore[method-assign]
    env.step = lambda *a, **k: frame  # type: ignore[method-assign]

    out = _runner_for(env, frame_skip=1).random_rollout(env, 40, seed=0)

    assert out["maps"] == [4, 5], out["maps"]
    assert len(out["act"]) == 40


def _menu_payload(scene="Scene_Menu"):
    return {"scene": scene, "messageBusy": False, "inGameplay": False,
            "menu": True, "tickerRunning": True, "awaitingChoice": False,
            "guardedChoice": False, "choices": []}


def test_cancel_presses_are_spaced_so_the_game_can_sample_them():
    """A release shorter than a frame is never observed.

    The game reads input once per 60 Hz frame and MZ triggers on a transition, so
    cancel presses dispatched back to back never let the key be seen UP - four of
    them registered as at most one. That is why the agent sat in Scene_Shop after
    "four" presses while a human leaves it with a single X.
    """
    import time as _time

    env = _env(_menu_payload())
    frame_ms = env.config.frame_ms
    presses = 3

    started = _time.perf_counter()
    out = env.escape_menu(max_presses=presses)
    elapsed_ms = (_time.perf_counter() - started) * 1000.0

    assert out["presses"] == presses, out
    # Each press holds the key for a frame, and every gap must be at least as long
    # as a frame or the release is invisible to the game.
    assert elapsed_ms >= presses * frame_ms + (presses - 1) * frame_ms, elapsed_ms


def test_the_escape_log_reports_the_outcome_not_the_attempt(caplog):
    """The line said "escaped" whether or not it had.

    A shop the agent could not leave therefore read as a shop it left and then
    re-entered, which sent the investigation after the wrong mechanism.
    """
    import logging

    env = _env(_menu_payload("Scene_Shop"))
    runner = _runner_for(env, frame_skip=1)
    with caplog.at_level(logging.WARNING):
        left = runner._leave_menu([env], 0)

    assert left is False
    assert "did not clear" in caplog.text, caplog.text
    assert "escaped a menu" not in caplog.text, caplog.text


MENU_SAFE = {"scene": "Scene_Menu", "messageBusy": False, "inGameplay": False,
             "tickerRunning": True, "awaitingChoice": False,
             "menuOperable": True, "guardedMenuEntry": False,
             "menuEntries": [{"panel": "_systemPanel", "index": 0,
                              "name": "STATIC_TEXT_MENU_SYSTEM_BACK_TOWN"}]}
MENU_DENIED = dict(MENU_SAFE, guardedMenuEntry=True,
                   menuEntries=[{"panel": "_systemPanel", "index": 4,
                                 "name": "STATIC_TEXT_MENU_SYSTEM_EXIT_GAME"}])
MENU_UNREADABLE = dict(MENU_SAFE, menuEntries=[])


def test_an_operable_menu_with_a_safe_entry_takes_input():
    """The shop, the item screens and the ESC menu are playable content.

    They used to be cancel-only, so the agent could walk into a shop and do
    nothing but leave.
    """
    assert _env(MENU_SAFE).unsafe_reason() is None


def test_the_last_column_is_refused_except_the_town_return():
    """The user's rule: everything in that column but "back to town" is one-way."""
    reason = _env(MENU_DENIED).unsafe_reason()
    assert reason and "must not be committed" in reason
    with pytest.raises(UnsafeSceneError):
        _env(MENU_DENIED).apply_action(0)


def test_an_operable_menu_whose_entry_cannot_be_read_is_refused():
    """Fail closed: an unclassified entry may be the one that costs the save."""
    reason = _env(MENU_UNREADABLE).unsafe_reason()
    assert reason and "could not be read" in reason


def test_a_menu_that_is_not_operable_is_still_cancel_only():
    payload = dict(MENU_SAFE, scene="Scene_File", menuOperable=False)
    reason = _env(payload).unsafe_reason()
    assert reason and "not Scene_Map gameplay" in reason


def test_the_shipped_agent_js_denies_the_whole_last_column_but_the_town():
    """The rule lives in JS, so it is loaded and executed, not re-implemented.

    The entry symbols were captured from the running game; the user's spoken
    wording named one of them "返回菜单" when the game's is BACK_TOWN.
    """
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    agent_js = __import__("pathlib").Path(__file__).resolve().parent.parent \
        / "src" / "ash" / "memory" / "agent.js"
    script = (
        "global.window = global;"
        "global.document = {addEventListener: function () {}};"
        "global.navigator = {};"
        "eval(require('fs').readFileSync(%r, 'utf8'));"
        "var V = window.__ash;"
        "var col = ['STATIC_TEXT_MENU_SYSTEM_BACK_TOWN',"
        " 'STATIC_TEXT_MENU_SYSTEM_RETURN_TO_TITLE',"
        " 'STATIC_TEXT_MENU_SYSTEM_RETURN_LOAD_GAME',"
        " 'STATIC_TEXT_MENU_SYSTEM_OPTIONS',"
        " 'STATIC_TEXT_MENU_SYSTEM_EXIT_GAME'];"
        "console.log(JSON.stringify({"
        "  guarded: col.map(function (n) { return V.isGuardedEntry([{panel: '_systemPanel', name: n}]); }),"
        "  item_category: V.isGuardedEntry([{panel: '_itemPanel',"
        "                     name: 'STATIC_TEXT_MENU_ITEM_ORNAMENTS'}]),"
        "  deny: V.MENU_ENTRY_DENY,"
        "  operable: V.OPERABLE_SCENES}));" % str(agent_js)
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-600:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["guarded"] == [False, True, True, True, True], got
    assert got["item_category"] is False, got
    assert "STATIC_TEXT_MENU_SYSTEM_BACK_TOWN" not in got["deny"], (
        "back to town is the one entry in that column the user allows")
    assert "Scene_Shop" in got["operable"], got
    # Only scenes the running game was SEEN to use are listed.  The classes
    # Scene_Item/Scene_Equip/Scene_Skill/Scene_Status do exist - enumerating
    # window.Scene_* shows them - but a screenshot of the menu showed items, orbs
    # and accessories drawn as PANELS inside Scene_Menu, and its "skill" panel is
    # a page that demonstrates the controls.  Listing unobserved scenes reads as
    # coverage without being any.
    for unobserved in ("Scene_Item", "Scene_Equip", "Scene_Skill", "Scene_Status"):
        assert unobserved not in got["operable"], unobserved
