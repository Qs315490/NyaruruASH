"""Engine state as the judge of whether an action did anything.

Pixels alone cannot separate "the map scrolled" from "the player took a step",
and a pair where the action had no consequence cannot teach an inverse model -
measured on the human recordings, 35% of pairs are that kind.  The engine state
is a training-time teacher (the delivered policy sees only pixels), which the
operator allowed explicitly, so these tests pin what it is allowed to claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from ash.data.effect import (
    PHYSICS_KEYS,
    UNUSABLE_POSITION_KEYS,
    describe_effects,
    own_delta,
)
from ash.env.fake_backend import FakeSpeedrunEnv
from ash.loop.bootstrap import _transition_motion
from ash.loop.runner import InferenceRunner
from ash.models.ash_policy import AshPolicy, AshPolicyConfig


def test_physics_position_is_the_authority():
    before = {"physics": dict(zip(PHYSICS_KEYS, (10.0, 5.0, 0.0, 0.0)))}
    after = {"physics": dict(zip(PHYSICS_KEYS, (13.0, 5.0, 1.5, 0.0)))}
    assert own_delta(before, after) == {"known": True, "moved": True, "source": "physics"}
    # Standing still is `known` and NOT moved: the two must not be conflated,
    # or an unreadable pair would silently count as a still one.
    assert own_delta(before, before) == {"known": True, "moved": False, "source": "physics"}


def test_frozen_tile_coordinates_are_never_used_as_movement():
    """The documented trap: in this game `x/y` and `_realX/_realY` never move.

    Reading them produced three wrong conclusions in this project, so a state
    that offers only those must be reported as unknown rather than as still -
    otherwise every real transition would look like "the player did not move".
    """
    frozen = {k: 7 for k in UNUSABLE_POSITION_KEYS}
    assert own_delta(frozen, frozen)["known"] is False
    assert own_delta(frozen, frozen)["moved"] is False


def test_velocity_source_covers_backends_without_world_position():
    """The fake backend reports velocity under `player`; the semantics match."""
    idle = {"player": {"vx": 0.0, "vy": 0.0}}
    assert own_delta(idle, idle) == {"known": True, "moved": False, "source": "velocity"}
    walking = {"player": {"vx": 2.0, "vy": 0.0}}
    d = own_delta(idle, walking)
    assert d["known"] is True and d["moved"] is True
    # Constant speed still counts as moving: velocity is an approximation for
    # backends that have no position, and a body at speed is not a still one.
    assert own_delta(walking, walking)["moved"] is True


def test_screen_position_is_the_fallback():
    a = {"player": {"screenX": 100.0, "screenY": 50.0}}
    b = {"player": {"screenX": 104.0, "screenY": 50.0}}
    assert own_delta(a, b)["moved"] is True
    assert own_delta(a, a)["moved"] is False


def test_unreadable_state_is_unknown_not_still():
    assert own_delta({"frame": 1}, {"frame": 2}) == {
        "known": False, "moved": False, "source": None}
    assert own_delta(None, None)["known"] is False


def test_describe_effects_counts_the_two_kinds_of_stillness():
    """`own_still` and `nothing_changed` are different questions.

    A player pressing into a wall is still on both counts; a player carried by a
    moving platform is still but the picture is not.  Collapsing them is exactly
    the confusion this module exists to remove.
    """
    frames = np.zeros((5, 4, 4, 3), dtype=np.uint8)
    frames[3:] = 255  # a visible change on the last transition only
    traj = {
        "obs": frames,
        "act": np.array([0, 1, 1, 0], dtype=np.int64),
        "state": [
            {"physics": {"px": 0.0, "py": 0.0, "vx": 0.0, "vy": 0.0}},
            {"physics": {"px": 0.0, "py": 0.0, "vx": 0.0, "vy": 0.0}},
            {"physics": {"px": 2.0, "py": 0.0, "vx": 1.0, "vy": 0.0}},
            {"physics": {"px": 2.0, "py": 0.0, "vx": 1.0, "vy": 0.0}},
            {"physics": {"px": 2.0, "py": 0.0, "vx": 1.0, "vy": 0.0}},
        ],
    }
    out = describe_effects([traj], motion=_transition_motion([traj]))
    assert out["transitions"] == 4
    assert out["own_moved"] == 1, "only the middle transition moves the player"
    assert out["own_still"] == 3
    assert out["unknown_state"] == 0
    assert out["noop_label"] == 2
    assert out["screen_changed"] == 1, "the picture changes on the last pair only"
    # Two pairs are still on both counts: the player did not move and neither did
    # the picture.  Those teach nothing and must be visible in the report.
    assert out["nothing_changed"] == 2
    assert out["share_nothing_changed"] == pytest.approx(0.5)
    assert out["share_own_moved"] == pytest.approx(0.25)
    assert len(out["per_step"]) == 4


def _runner(env):
    """A tiny runner wired to the fake env, with a K that is fit but irrelevant.

    Built here rather than imported from another test file: this suite keeps its
    files self-contained, and the effect report does not depend on K at all.
    """
    from ash.memory.kdm import KeyMomentModel

    rng = np.random.default_rng(3)
    embs, ids = [], []
    for c in range(4):
        embs.append(rng.normal(size=8) * 3 + rng.normal(size=(24, 8)) * 0.1)
        ids.extend([f"v{c}"] * 24)
    kdm = KeyMomentModel(min_cluster_size=5, min_distinct_trajectories=2)
    kdm.fit(np.concatenate(embs).astype(np.float32), ids)

    class _Embedder:
        def embed(self, frames, batch_size=64):
            frames = np.asarray(frames)
            out = np.empty((len(frames), 8), dtype=np.float32)
            for i, f in enumerate(frames):
                seed = int(np.asarray(f, dtype=np.uint8).astype(np.int64).sum()) % (2**31)
                out[i] = np.random.default_rng(seed).normal(size=8)
            return out

    kdm.embedder = _Embedder()
    space = env.action_space
    policy = AshPolicy(AshPolicyConfig(image_size=32, w_s=4, w_l=2, num_layers=1,
                                       num_actions=len(space)))
    return InferenceRunner(lambda: env, policy, kdm, action_space=space, w_s=4, w_l=2,
                           image_size=32, device="cpu", key_moment_cooldown=0)


def test_runner_carries_engine_state_alongside_the_frames():
    """Regression: `_frame()` used to drop `obs.state` on the floor.

    Without this, every transition looks "unknown" to the effect report and the
    teacher is silently absent - a failure that looks like a working run.
    """
    env = FakeSpeedrunEnv()
    out = _runner(env).run([env], delta=6, timeout_s=30, random_steps=5)
    for key in ("trajectories", "random_trajectories"):
        for traj in out[key]:
            states = traj["state"]
            assert len(states) == len(traj["obs"]), "one state per frame"
            assert isinstance(states[0], dict) and states[0], "state must not be empty"
    both = list(out["trajectories"]) + list(out["random_trajectories"])
    effects = describe_effects(both, motion=_transition_motion(both))
    assert effects["transitions"] > 0
    assert effects["unknown_state"] == 0, "the fake backend reports readable state"
    assert effects["own_moved"] > 0, "random actions do move the fake character"


def test_action_intent_comes_from_the_buttons_not_a_class_table():
    """Deriving intent from the mask keeps it right when the space grows."""
    from ash.actions.space import ActionSpace
    from ash.data.effect import action_intent

    space = ActionSpace.minimal()
    by_name = {", ".join(__import__("ash.actions.space", fromlist=["buttons_from_mask"])
                         .buttons_from_mask(m)) or "noop": action_intent(m)
               for m in space.masks}
    assert by_name["noop"] == (0, 0)
    assert by_name["left"] == (-1, 0)
    assert by_name["right"] == (1, 0)
    # py grows downward, and jump pushes against it - written down once because
    # getting the sign backwards silently inverts every judgement here.
    assert by_name["jump"] == (0, -1)
    assert by_name["down"] == (0, 1)
    assert by_name["right, jump"] == (1, -1)
    assert by_name["left, attack"] == (-1, 0)


def test_effective_requires_going_the_way_the_action_pushes():
    """The stricter gate: measured 57% of intent-bearing transitions pass."""
    from ash.actions.space import ActionSpace
    from ash.data.effect import is_effective

    space = ActionSpace.minimal()
    left, right, jump = space.mask_at(1), space.mask_at(2), space.mask_at(3)
    start = {"physics": {"px": 100.0, "py": 100.0}}

    def at(px, py):
        return {"physics": {"px": float(px), "py": float(py)}}

    assert is_effective(start, at(60, 100), left)["effective"] is True
    assert is_effective(start, at(60, 100), right)["effective"] is False, "wrong way"
    assert is_effective(start, at(99, 100), left)["effective"] is False, (
        "one pixel is blocked, not effective - this is the 43% the gate removes")
    assert is_effective(start, at(100, 50), jump)["effective"] is True
    assert is_effective(start, at(100, 150), jump)["effective"] is False


def test_actions_without_displacement_intent_cannot_be_judged():
    """noop must not be called effective, and this is not pedantry.

    A noop step during which gravity moved the player is not evidence about
    noop: counting it taught the labeler that a changing picture means no key
    was pressed, which is the confound the whole investigation ran into.
    """
    from ash.actions.space import ActionSpace
    from ash.data.effect import is_effective

    space = ActionSpace.minimal()
    noop = space.mask_at(0)
    start = {"physics": {"px": 100.0, "py": 100.0}}
    fell = {"physics": {"px": 100.0, "py": 196.0}}
    assert is_effective(start, start, noop)["effective"] is None
    assert is_effective(start, fell, noop)["effective"] is None
    assert is_effective(start, fell, space.mask_at(6))["effective"] is None, "attack"
    assert is_effective({"x": 1}, {"x": 2}, space.mask_at(1))["effective"] is None, (
        "unreadable state is undecided, never effective")
