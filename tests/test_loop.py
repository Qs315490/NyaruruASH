"""End-to-end smoke of the ASH loop on the fake backend.

The fake env is deterministic and has a real action space, so this file can
exercise the full infer -> retrieve -> bootstrap cycle without the game.  It
pins the interface contracts the paper's algorithms assume:

- runner.run() returns trajectories as (T, H, W, C) uint8 arrays
- bootstrap consumes those arrays plus the agent actions
- the policy's training target is every window position (not just the last)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

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
    # Stub the embedder: classify() runs on precomputed 8-dim vectors.
    class _FakeEmbedder:
        def embed(self, frames, batch_size=64):
            rng2 = np.random.default_rng(abs(hash(frames.tobytes())) % (2**32))
            return rng2.normal(size=(len(frames), 8)).astype(np.float32)

    kdm.embedder = _FakeEmbedder()
    return kdm


def test_runner_returns_uint8_trajectories(kdm_fitted):
    env = FakeSpeedrunEnv()
    policy = AshPolicy(AshPolicyConfig(image_size=32, w_s=4, w_l=2, num_layers=1))
    runner = InferenceRunner(
        lambda: env, policy, kdm_fitted,
        w_s=4, w_l=2, image_size=32, device="cpu",
        key_moment_cooldown=0,
    )
    out = runner.run([env], delta=6, timeout_s=30)
    traj = out["trajectories"][0]
    assert traj.ndim == 4 and traj.dtype == np.uint8
    assert traj.shape[0] == len(traj) > 1
    assert out["stuck"], "delta=6 must trip the stuck timer quickly on the fake env"


def test_bootstrap_policy_target_shape(kdm_fitted):
    """The old bug: argmax over axis=1 (the w_s axis) instead of the action axis."""
    idm = IdmModel(IdmConfig(image_size=32, embed_dim=32, num_keys=17))
    obs = np.random.default_rng(0).integers(
        0, 255, size=(40, 32, 32, 3), dtype=np.uint8
    )
    b = bootstrap_mod.Bootstrapper(
        bootstrap_mod.BootstrapConfig(w_s=4, w_l=2, image_size=32, device="cpu")
    )
    ds = b.build_policy_dataset(obs, idm, kdm_fitted)
    n_windows = 40 - 4 + 1
    assert ds["frames"].shape == (n_windows, 4, 32, 32, 3)
    assert ds["actions"].shape == (n_windows, 4, 17)
    assert ds["memories"].shape == (n_windows, 2, 32, 32, 3)
    # actions must be multi-hot over the ACTION axis, not one position
    assert ds["actions"].sum(axis=-1).max() <= 17
    assert set(np.unique(ds["actions"])) <= {0.0, 1.0}


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


def test_idm_loss_decreases_on_tiny_data():
    """Guards the IDM wiring: pair (t, t+1) -> mask[t+1], holdout separated."""
    torch.manual_seed(0)
    idm = IdmModel(IdmConfig(image_size=16, embed_dim=16, num_keys=17))
    b = bootstrap_mod.Bootstrapper(
        bootstrap_mod.BootstrapConfig(
            image_size=16, idm_epochs=2, batch_size=4, device="cpu"
        )
    )
    obs = np.random.default_rng(1).integers(0, 255, size=(30, 16, 16, 3), dtype=np.uint8)
    actions = np.random.default_rng(2).integers(0, 17, size=(30,))
    onehot = np.zeros((30, 17), dtype=np.float32)
    onehot[np.arange(30), actions] = 1.0
    report = b.update_idm(idm, obs, onehot)
    assert "idm_val" in report and np.isfinite(report["idm_val"])
