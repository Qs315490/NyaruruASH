"""A round must actually update pi once the IDM is informative.

Measured: with an IDM pretrained on the recorded human demonstrations
(`ash pretrain-idm`) plus `--idm-replay`, the pseudo-labels stop collapsing, the
`max_pseudo_majority` guard stops refusing, and the round enters `update_policy`
- a step that had never executed before, because every previous round was refused
there.  It then ran unbounded: 3126 optimizer steps of 8x64 frames each for one
9405-frame video, with nothing logged until an epoch ended, so a round looked
hung for the better part of an hour.

These tests pin: pi is updated, and the step cap bounds the work.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ash.actions.space import ActionSpace  # noqa: E402
from ash.loop.bootstrap import BootstrapConfig, Bootstrapper  # noqa: E402
from ash.models.ash_policy import AshPolicy, AshPolicyConfig  # noqa: E402
from ash.models.idm import IdmConfig, IdmModel  # noqa: E402


class _InformativeIdm(torch.nn.Module):
    """Logits that depend on the input, so the labels are not one class.

    A stub returning a constant is exactly the degenerate case the guard refuses,
    so it cannot exercise the update path at all.
    """

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.config = IdmConfig(image_size=16, embed_dim=16, num_actions=num_actions)
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.zeros(num_actions))

    def forward(self, frame_a, frame_b):
        # Frame-indexed "evidence": the mean pixel picks the class, so consecutive
        # pairs get varied labels instead of one constant.
        evidence = frame_a.mean(dim=(1, 2, 3))
        return self.bias + self.scale * torch.sin(evidence).unsqueeze(1) * 10.0


class _KStub:
    class _Emb:
        def embed(self, frames, batch_size=64):
            return np.zeros((len(frames), 4), dtype=np.float32)

    embedder = _Emb()

    def observe(self, embedding, seen):
        return False, -1, False

    def classify_sequence(self, embeddings):
        return np.zeros(len(embeddings), dtype=bool)

    def fit(self, embeddings, trajectory_ids):
        return {"clusters_total": 0, "clusters_kept": 0, "noise_rate": 1.0}


def _run(tmp_path, frames, actions, *, max_policy_steps):
    space = ActionSpace.minimal()
    b = Bootstrapper(BootstrapConfig(
        image_size=16, w_s=4, w_l=2, idm_epochs=1, policy_epochs=3,
        batch_size=4, device="cpu", max_policy_steps=max_policy_steps,
        max_pseudo_majority=1.01,   # accept the labels; the guard is tested elsewhere
    ))
    policy = AshPolicy(AshPolicyConfig(image_size=16, w_s=4, w_l=2, num_layers=1,
                                       num_actions=len(space)))
    idm = _InformativeIdm(len(space))

    def loader(ids=None):
        yield "video", frames

    traj = {"obs": frames, "act": np.asarray(actions[: len(frames) - 1], dtype=np.int64)}
    return b.run(policy, idm, _KStub(), [traj], loader, tmp_path,
                 retrieved_ids=["video"], random_trajectories=[])


def _fixture(rng, n=40):
    frames = rng.integers(0, 255, (n, 16, 16, 3), dtype=np.uint8)
    actions = rng.integers(0, 20, n)
    return frames, actions


def test_pi_is_updated_and_the_step_cap_bounds_it(tmp_path):
    rng = np.random.default_rng(0)
    frames, actions = _fixture(rng)
    report = _run(tmp_path, frames, actions, max_policy_steps=2)
    updates = report.get("policy")
    assert updates, "pi was not updated once the pseudo-labels stopped collapsing"
    policy = updates[0]          # one entry per corpus video
    assert policy["policy_steps"] == 2, (
        "the cap must bound the optimizer steps: %r" % policy
    )
    assert np.isfinite(policy["policy_val"])


def test_zero_means_unbounded(tmp_path):
    rng = np.random.default_rng(0)
    frames, actions = _fixture(rng)
    report = _run(tmp_path, frames, actions, max_policy_steps=0)
    assert report["policy"][0]["policy_steps"] > 2
