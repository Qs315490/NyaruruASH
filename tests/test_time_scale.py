"""One time scale for the whole pipeline.

The IDM is trained on the agent's own (obs[t], obs[t+1]) pairs and then applied
to corpus frames.  Those two must be the same Δt or the IDM is asked to explain
a change it has never seen.  That is exactly what happened: the agent stepped
one game frame (1/60 s) while the corpus held frames 2 s apart, a 120x
mismatch.  The IDM answered with a single class for 99.8% of corpus frames and
the policy "converged" by predicting that class - with nothing in the logs
looking wrong.

`config/game.yaml:control_interval_s` is now the one number both sides derive
from, and these tests pin that they stay derived.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ash.actions.space import ActionSpace  # noqa: E402
from ash.cli.main import _check_corpus_interval  # noqa: E402
from ash.config import GameConfig, load_game_config  # noqa: E402
from ash.env.fake_backend import FakeSpeedrunEnv  # noqa: E402
from ash.loop.runner import InferenceRunner  # noqa: E402
from ash.models.ash_policy import AshPolicy, AshPolicyConfig  # noqa: E402


class _NoKeyMoments:
    """A K that never fires, so the runner's loop can be tested without HDBSCAN.

    It also documents the two calls the runner makes: embed the observation,
    then ask observe().  Returning "no" keeps the episode running instead of
    resetting the stuck timer, which is what a working-but-unmatched K looks
    like.
    """

    class _Emb:
        def embed(self, frames, batch_size=64):
            return np.zeros((len(frames), 4), dtype=np.float32)

    embedder = _Emb()

    def observe(self, embedding, seen_clusters):
        return False, -1, False

    def classify_sequence(self, embeddings):
        return np.zeros(len(embeddings), dtype=bool)

    def fit(self, embeddings, trajectory_ids):
        return {"clusters_total": 0, "clusters_kept": 0, "noise_rate": 1.0}


def _runner(env, **kw):
    policy = AshPolicy(AshPolicyConfig(image_size=32, w_s=4, w_l=2, num_layers=1))
    return InferenceRunner(
        lambda: env, policy, _NoKeyMoments(),
        action_space=env.action_space, w_s=4, w_l=2, image_size=32, device="cpu",
        **kw,
    )


def test_control_interval_is_the_single_source():
    """Both sides must be derived, never written down twice."""
    game = GameConfig(fps=60, control_interval_s=0.25)
    assert game.control_frame_skip == 15          # 0.25 s * 60 fps
    assert game.corpus_fps == pytest.approx(4.0)  # 1 / 0.25 s
    # The two must describe the same Δt.
    assert game.control_frame_skip / game.fps == pytest.approx(game.control_interval_s)
    assert 1.0 / game.corpus_fps == pytest.approx(game.control_interval_s)


def test_shipped_config_follows_the_paper():
    """The paper's Appendix G sets the environment timestep to 0.25 s."""
    game = load_game_config()
    assert game.control_interval_s == pytest.approx(0.25)
    assert game.control_frame_skip == 15
    assert game.corpus_fps == pytest.approx(4.0)


def test_runner_advances_the_control_interval_per_action():
    """One action must advance frame_skip game frames - not one."""
    env = FakeSpeedrunEnv()
    seen: list[int] = []
    real_step = env.step

    def spy(action, frames=1, **kw):
        seen.append(frames)
        return real_step(action, frames=frames, **kw)

    env.step = spy  # type: ignore[method-assign]
    _runner(env, frame_skip=15).run([env], delta=3, timeout_s=5)

    assert seen, "the round must have stepped at least once"
    assert set(seen) == {15}, "every action must advance the full interval"


def test_corpus_interval_guard_catches_a_mismatch(tmp_path):
    """A corpus sampled at another rate must be refused, not silently used."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "v", "frames": 10, "fps": 0.5}]))
    reason = _check_corpus_interval(str(tmp_path), expected_fps=4.0)
    assert reason and "0.5" in reason and "4" in reason


def test_corpus_interval_guard_accepts_a_match(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "v", "frames": 10, "fps": 4.0}]))
    assert _check_corpus_interval(str(tmp_path), expected_fps=4.0) is None


def test_corpus_interval_guard_ignores_a_manifest_without_fps(tmp_path):
    """The live smoke corpus records no fps; that is not a mismatch."""
    (tmp_path / "manifest.json").write_text(json.dumps([{"id": "v", "frames": 10}]))
    assert _check_corpus_interval(str(tmp_path), expected_fps=4.0) is None


def test_random_rollout_produces_transitions_at_the_control_interval():
    env = FakeSpeedrunEnv()
    seen: list[int] = []
    real_step = env.step

    def spy(action, frames=1, **kw):
        seen.append(frames)
        return real_step(action, frames=frames, **kw)

    env.step = spy  # type: ignore[method-assign]
    traj = _runner(env, frame_skip=15).random_rollout(env, 6)

    assert len(traj["act"]) == 6
    assert traj["obs"].shape[0] == 7, "one more frame than actions"
    assert set(seen) == {15}


def test_random_rollout_uses_a_spread_of_actions():
    """It exists to widen the IDM's coverage, so it must not be one button."""
    traj = _runner(FakeSpeedrunEnv(), frame_skip=1).random_rollout(FakeSpeedrunEnv(), 64)
    assert len(set(traj["act"].tolist())) > 1


def test_random_rollout_obeys_the_safety_gate():
    """Random keystrokes on a menu are still menu selections."""
    env = FakeSpeedrunEnv()
    env.step = lambda *a, **k: pytest.fail("must not press a key on a menu")  # type: ignore[method-assign]
    env.unsafe_reason = lambda: "scene 'Scene_Title' is not Scene_Map gameplay"  # type: ignore[method-assign]

    traj = _runner(env, frame_skip=1).random_rollout(env, 10)
    assert len(traj["act"]) == 0


def test_run_surfaces_random_trajectories_for_the_idm():
    """The paper's supplement must actually reach the bootstrap."""
    env = FakeSpeedrunEnv()
    out = _runner(env, frame_skip=4).run([env], delta=3, timeout_s=5, random_steps=5)

    assert len(out["random_trajectories"]) == 1
    assert len(out["random_trajectories"][0]["act"]) == 5


def test_run_without_random_steps_reports_none():
    env = FakeSpeedrunEnv()
    out = _runner(env, frame_skip=1).run([env], delta=3, timeout_s=5, random_steps=0)
    assert out["random_trajectories"] == []


def test_bootstrap_counts_agent_and_random_transitions(tmp_path):
    """The IDM report must show the supplement, so a silent regression shows."""
    from ash.loop.bootstrap import Bootstrapper, BootstrapConfig

    space = ActionSpace.minimal()
    from ash.models.ash_policy import AshPolicy as P
    from ash.models.idm import IdmModel, IdmConfig

    b = Bootstrapper(BootstrapConfig(
        image_size=16, w_s=4, w_l=2, idm_epochs=1, policy_epochs=1,
        batch_size=4, device="cpu",
    ))
    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=len(space)))
    policy = P(AshPolicyConfig(image_size=16, w_s=4, w_l=2, num_layers=1,
                               num_actions=len(space)))
    rng = np.random.default_rng(0)
    agent = {"obs": rng.integers(0, 255, (8, 16, 16, 3), dtype=np.uint8),
             "act": rng.integers(0, len(space), 7)}
    random_traj = {"obs": rng.integers(0, 255, (6, 16, 16, 3), dtype=np.uint8),
                   "act": rng.integers(0, len(space), 5)}

    def loader(ids=None):
        yield "v", rng.integers(0, 255, (10, 16, 16, 3), dtype=np.uint8)

    report = b.run(policy, idm, _NoKeyMoments(), [agent], loader, tmp_path,
                   random_trajectories=[random_traj])

    assert report["idm"]["agent_transitions"] == 7
    assert report["idm"]["random_transitions"] == 5


def test_random_rollout_varies_between_rounds():
    """The supplement must explore something new each round.

    A hard-coded seed replayed the identical action sequence in every round, so
    the "random-policy samples" the paper adds to the IDM's training set never
    widened beyond the same hundred steps.
    """
    env = FakeSpeedrunEnv()
    runner = _runner(env, frame_skip=4)
    a = runner.random_rollout(env, 12, seed=0)["act"]
    b = runner.random_rollout(env, 12, seed=1)["act"]
    c = runner.random_rollout(env, 12, seed=0)["act"]
    assert len(a) == len(b) == 12
    assert not np.array_equal(a, b), "a new round must explore differently"
    assert np.array_equal(a, c), "the same round must replay identically"
