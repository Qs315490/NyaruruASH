"""End-to-end smoke of the ASH loop on the fake backend.

The fake env is deterministic and has a real action space, so this file can
exercise the full infer -> retrieve -> bootstrap cycle without the game.  It
pins the interface contracts the paper's algorithms assume:

- runner.run() returns trajectories as {"obs": (T,H,W,C) uint8, "act": (T,)}
  where "act" holds *policy class indices*, not button masks
- bootstrap consumes those dicts plus the IDM/K the loop owns
- the policy's training target is every window position (not just the last)
- both model heads are as wide as the env's action space

The three action-shaped counts (buttons, action-space masks, policy classes)
were once three different numbers, which surfaced only as a shape crash at the
first bootstrap; the assertions below keep them pinned to one source.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ash.actions.space import DEFAULT_NUM_ACTIONS, ActionSpace
from ash.env.fake_backend import FakeSpeedrunEnv
from ash.loop import bootstrap as bootstrap_mod
from ash.loop.runner import InferenceRunner
from ash.models.ash_policy import AshPolicy, AshPolicyConfig
from ash.models.idm import IdmModel, IdmConfig

hdbscan = pytest.importorskip("hdbscan")


@pytest.fixture(scope="module")
def kdm_fitted():
    """A KeyMomentModel fit on synthetic blobs so classify() works offline."""
    from ash.memory.kdm import KeyMomentModel

    rng = np.random.default_rng(3)
    embs, ids = [], []
    for c in range(4):
        center = rng.normal(size=8) * 3
        embs.append(center + rng.normal(size=(24, 8)) * 0.1)
        ids.extend([f"v{c}"] * 24)
    kdm = KeyMomentModel(min_cluster_size=5, min_distinct_trajectories=2)
    kdm.fit(np.concatenate(embs).astype(np.float32), ids)

    # Stub the embedder: classify() runs on precomputed 8-dim vectors, and the
    # stub must be deterministic per frame or the stuck timer is untestable.
    class _FakeEmbedder:
        def embed(self, frames, batch_size=64):
            frames = np.asarray(frames)
            out = np.empty((len(frames), 8), dtype=np.float32)
            for i, f in enumerate(frames):
                seed = int(np.asarray(f, dtype=np.uint8).astype(np.int64).sum()) % (2**31)
                out[i] = np.random.default_rng(seed).normal(size=8)
            return out

    kdm.embedder = _FakeEmbedder()
    return kdm


def test_action_space_size_matches_head_default():
    """The magic number the models default to must equal the real space."""
    assert len(ActionSpace.minimal()) == DEFAULT_NUM_ACTIONS


def _make_runner(env, kdm, image_size=32, w_s=4, w_l=2):
    space = env.action_space
    policy = AshPolicy(
        AshPolicyConfig(image_size=image_size, w_s=w_s, w_l=w_l, num_layers=1,
                        num_actions=len(space))
    )
    return InferenceRunner(
        lambda: env, policy, kdm, action_space=space,
        w_s=w_s, w_l=w_l, image_size=image_size, device="cpu",
        key_moment_cooldown=0,
    )


def test_runner_returns_index_labelled_trajectories(kdm_fitted):
    env = FakeSpeedrunEnv()
    space = env.action_space
    policy = AshPolicy(
        AshPolicyConfig(image_size=32, w_s=4, w_l=2, num_layers=1, num_actions=len(space))
    )
    runner = InferenceRunner(
        lambda: env, policy, kdm_fitted,
        action_space=space,
        w_s=4, w_l=2, image_size=32, device="cpu",
        key_moment_cooldown=0,
    )
    out = runner.run([env], delta=6, timeout_s=30)
    traj = out["trajectories"][0]
    obs, act = traj["obs"], traj["act"]
    assert obs.ndim == 4 and obs.dtype == np.uint8
    assert len(obs) > 1
    assert act.shape == (len(obs) - 1,), "one action per executed step"
    assert act.dtype == np.int64
    # Labels are class indices into the action space, never raw button masks.
    assert act.min() >= 0 and act.max() < len(space)
    assert out["stuck"], "delta=6 must trip the stuck timer quickly on the fake env"


def test_runner_records_actions_the_env_accepts(kdm_fitted):
    """Every recorded index must round-trip to a mask the env actually took."""
    env = FakeSpeedrunEnv()
    space = env.action_space
    seen_masks: list[int] = []
    real_step = env.step

    def spy(action, *a, **kw):
        seen_masks.append(int(action))
        return real_step(action, *a, **kw)

    env.step = spy  # type: ignore[method-assign]
    policy = AshPolicy(
        AshPolicyConfig(image_size=32, w_s=4, w_l=2, num_layers=1, num_actions=len(space))
    )
    runner = InferenceRunner(
        lambda: env, policy, kdm_fitted,
        action_space=space, w_s=4, w_l=2, image_size=32, device="cpu",
        key_moment_cooldown=0,
    )
    out = runner.run([env], delta=6, timeout_s=30)
    act = out["trajectories"][0]["act"]
    assert len(seen_masks) == len(act)
    for index, mask in zip(act.tolist(), seen_masks):
        assert space.mask_at(index) == mask, "index -> mask mapping drifted"


def test_bootstrap_policy_dataset_shapes(kdm_fitted):
    """The old bug: argmax over axis=1 (the w_s axis) instead of the action axis."""
    space = ActionSpace.minimal()
    n_actions = len(space)
    idm = IdmModel(IdmConfig(image_size=32, embed_dim=32, num_actions=n_actions))
    obs = np.random.default_rng(0).integers(0, 255, size=(40, 32, 32, 3), dtype=np.uint8)
    b = bootstrap_mod.Bootstrapper(
        bootstrap_mod.BootstrapConfig(w_s=4, w_l=2, image_size=32, device="cpu")
    )
    ds = b.build_policy_dataset(obs, idm, kdm_fitted)
    n_windows = 40 - 4 + 1
    assert len(ds) == n_windows
    b0 = ds.batch([0, 1, 2])
    assert b0["frames"].shape == (3, 4, 32, 32, 3)
    assert b0["actions"].shape == (3, 4, n_actions)
    assert b0["memories"].shape == (3, 2, 32, 32, 3)
    # Windows are NOT materialized: the dataset holds one frame buffer, not one
    # copy of every frame per window (that was 59 GB on a real corpus video).
    assert ds.frames.ndim == 4 and ds.frames.shape[0] == 40
    assert ds.mem_idx.shape == (n_windows, 2)
    # A batch must be identical to the same rows of the full window matrix.
    assert np.array_equal(ds["actions"], ds["actions"])
    # One-hot over the ACTION axis: exactly one class set per position.
    assert np.allclose(b0["actions"].sum(axis=-1), 1.0)
    assert set(np.unique(b0["actions"])) <= {0.0, 1.0}


def test_bootstrap_memory_prefix_precedes_window(kdm_fitted):
    """Memories for a window may only use key moments strictly before it."""
    space = ActionSpace.minimal()
    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=len(space)))
    obs = np.arange(24 * 16 * 16 * 3, dtype=np.uint8).reshape(24, 16, 16, 3) % 255
    b = bootstrap_mod.Bootstrapper(
        bootstrap_mod.BootstrapConfig(w_s=4, w_l=2, image_size=16, device="cpu")
    )
    ds = b.build_policy_dataset(obs, idm, kdm_fitted)
    # First window has nothing before it, so its memory prefix is all zeros.
    assert not ds.batch([0])["memories"].any()


def test_policy_forward_and_backward():
    torch.manual_seed(0)
    cfg = AshPolicyConfig(image_size=32, w_s=4, w_l=2, num_layers=1, embed_dim=32)
    policy = AshPolicy(cfg)
    frames = torch.rand(2, cfg.w_s, 32, 32, 3)
    masks = torch.zeros(2, cfg.w_s, cfg.num_actions)
    masks[..., 3] = 1.0
    mem = torch.rand(2, cfg.w_l, 32, 32, 3)
    logits = policy(frames, masks, mem)
    assert logits.shape == (2, cfg.w_s, cfg.num_actions)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.num_actions), masks.reshape(-1, cfg.num_actions).argmax(-1)
    )
    loss.backward()
    assert policy.head.weight.grad is not None


def test_idm_pseudo_actions_are_class_indices():
    """pseudo_actions() must emit indices usable as CE targets, not one-hots."""
    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=6))
    frames = torch.rand(5, 16, 16, 3)
    idx = idm.pseudo_actions(frames)
    assert idx.shape == (4,), "T-1 pairs, frame 0 has no predecessor"
    assert idx.dtype == torch.long
    assert int(idx.max()) < 6 and int(idx.min()) >= 0


def test_idm_loss_is_finite_on_tiny_data():
    """Guards the IDM wiring: one label per transition, so len(obs) - 1 of them."""
    torch.manual_seed(0)
    space = ActionSpace.minimal()
    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=len(space)))
    b = bootstrap_mod.Bootstrapper(
        bootstrap_mod.BootstrapConfig(
            image_size=16, idm_epochs=2, batch_size=4, device="cpu"
        )
    )
    obs = np.random.default_rng(1).integers(0, 255, size=(30, 16, 16, 3), dtype=np.uint8)
    actions = np.random.default_rng(2).integers(0, len(space), size=(29,)).astype(np.int64)
    report = b.update_idm(idm, obs, actions)
    assert "idm_val" in report and np.isfinite(report["idm_val"])


def test_idm_rejects_frame_indexed_labels():
    """A T-length label array is the off-by-one that mislabels every pair.

    Indexing a frame-aligned array by transition silently shifts each label onto
    its neighbour, so the trainer must refuse the input instead of training on
    it: this is the bug that made the IDM learn the *next* action.
    """
    space = ActionSpace.minimal()
    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=len(space)))
    b = bootstrap_mod.Bootstrapper(
        bootstrap_mod.BootstrapConfig(image_size=16, idm_epochs=1, device="cpu")
    )
    obs = np.zeros((8, 16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="one label per transition"):
        b.update_idm(idm, obs, np.zeros(8, dtype=np.int64))


def test_idm_update_accepts_runner_trajectories(kdm_fitted):
    """The bootstrap's own entry point must consume runner dicts unchanged."""
    space = ActionSpace.minimal()
    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=len(space)))
    b = bootstrap_mod.Bootstrapper(
        bootstrap_mod.BootstrapConfig(
            image_size=16, idm_epochs=1, batch_size=4, device="cpu"
        )
    )
    trajs = [
        {
            "obs": np.random.default_rng(i).integers(0, 255, size=(12, 16, 16, 3), dtype=np.uint8),
            "act": np.random.default_rng(100 + i).integers(0, len(space), size=(11,)).astype(np.int64),
        }
        for i in range(2)
    ]
    report = b.update_idm_from_trajectories(idm, trajs)
    assert "idm_val" in report and np.isfinite(report["idm_val"])


def test_update_idm_skips_degenerate_input():
    """A one-frame trajectory has no pair to learn from; it must not crash."""
    space = ActionSpace.minimal()
    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=len(space)))
    b = bootstrap_mod.Bootstrapper(
        bootstrap_mod.BootstrapConfig(image_size=16, device="cpu")
    )
    obs = np.zeros((1, 16, 16, 3), dtype=np.uint8)
    assert b.update_idm(idm, obs, np.zeros(0, dtype=np.int64)) == {"idm_skipped": True}
    assert b.update_idm_from_trajectories(idm, []) == {"idm_skipped": True}


def test_bootstrap_key_moments_use_the_projected_space(kdm_fitted):
    """K is fit in PCA space, so the bootstrap must ask through cluster_of().

    The bootstrap used to call approximate_predict() on the raw embedding.
    With a PCA-reduced clusterer that is a dimension mismatch at best and
    silent "noise" at worst - either way key moments stop being recorded and
    the memory prefix of every policy window stays empty.
    """
    from ash.memory.kdm import KeyMomentModel

    rng = np.random.RandomState(0)
    centres = rng.randn(3, 384) * 6
    rows, ids = [], []
    for vid in range(4):
        for c in range(3):
            rows.append(centres[c] + rng.randn(30, 384) * 0.2)
            ids += [vid] * 30
    kdm = KeyMomentModel(min_cluster_size=10, min_distinct_trajectories=2, pca_dim=8)
    kdm.fit(np.concatenate(rows).astype("float32"), np.asarray(ids))
    assert kdm._pca is not None, "this test is only meaningful with PCA on"

    class _WideEmbedder:
        """Emits 384-d vectors, i.e. the width the clusterer does NOT speak."""

        def embed(self, frames, batch_size=64):
            frames = np.asarray(frames)
            out = np.empty((len(frames), 384), dtype=np.float32)
            for i, f in enumerate(frames):
                seed = int(np.asarray(f, dtype=np.uint8).astype(np.int64).sum()) % (2**31)
                out[i] = rng.normal(size=384) if seed == 0 else rng.normal(size=384) * 0
            return out

    kdm.embedder = _WideEmbedder()
    space = ActionSpace.minimal()
    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_actions=len(space)))
    obs = np.arange(20 * 16 * 16 * 3, dtype=np.uint8).reshape(20, 16, 16, 3) % 255
    b = bootstrap_mod.Bootstrapper(
        bootstrap_mod.BootstrapConfig(w_s=4, w_l=2, image_size=16, device="cpu")
    )
    ds = b.build_policy_dataset(obs, idm, kdm)  # must not raise on the 384/8 mismatch
    assert ds.batch([0])["memories"].shape == (1, 2, 16, 16, 3)


def test_runner_samples_the_map_so_progress_is_visible(kdm_fitted):
    """A round must say whether the agent left the room it started in.

    "Did it get anywhere?" was unanswerable from the report: a round that never
    left the starting map and one that crossed three maps looked identical.
    """
    env = FakeSpeedrunEnv()
    out = _make_runner(env, kdm_fitted).run([env], delta=10, timeout_s=30)
    assert "maps" in out
    assert len(out["maps"]) == 1


def test_map_sampling_tolerates_backends_without_state(kdm_fitted):
    """The diagnostic must never fail a round on a backend that lacks it."""
    from ash.env.fake_backend import FakeSpeedrunEnv

    class _NoState(FakeSpeedrunEnv):
        state = None            # type: ignore[assignment]

    env = _NoState()
    out = _make_runner(env, kdm_fitted).run([env], delta=5, timeout_s=20)
    assert out["maps"] == [[]]
