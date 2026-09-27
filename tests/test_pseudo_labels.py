"""Refuse to train pi on constant pseudo-labels.

A bias-dominated IDM answers every corpus pair with the same class.  Training
pi on that teaches it to emit one constant action, and the resulting loss
(policy_val ~1e-6) reads like convergence.  Measured on the real checkpoints:
per-frame logit variance 0.008 against a class prior of 0.13, so the argmax
never moved with the input - and a *randomly initialised* IDM already agreed
with itself 97.2% of the time.

The pipeline must therefore measure the label distribution before updating pi
and skip the update when it is degenerate, which is what these tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ash.actions.space import ActionSpace  # noqa: E402
from ash.loop.bootstrap import (  # noqa: E402
    Bootstrapper,
    BootstrapConfig,
    pseudo_label_stats,
)
from ash.models.ash_policy import AshPolicy, AshPolicyConfig  # noqa: E402
from ash.models.idm import IdmConfig  # noqa: E402

N_ACTIONS = len(ActionSpace.minimal())


class _K:
    """No key moments; only the two calls build_policy_dataset makes."""

    class _Emb:
        def embed(self, frames, batch_size=64):
            return np.zeros((len(frames), 4), dtype=np.float32)

    embedder = _Emb()

    def classify(self, embedding, seen_clusters):
        return False

    def classify_sequence(self, embeddings):
        return np.zeros(len(embeddings), dtype=bool)

    def cluster_of(self, embedding):
        return -1

    def fit(self, embeddings, trajectory_ids):
        return {"clusters_total": 0, "clusters_kept": 0, "noise_rate": 1.0}


class _ConstantIdm(torch.nn.Module):
    """Answers every pair with one class - the failure being guarded against."""

    def __init__(self, num_actions: int = N_ACTIONS, cls: int = 3) -> None:
        super().__init__()
        # a real IdmConfig: save_idm() serialises it with as_dict()
        self.config = IdmConfig(image_size=16, embed_dim=16, num_actions=num_actions)
        self.cls = cls
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, frame_a, frame_b):
        logits = torch.zeros(frame_a.shape[0], self.config.num_actions)
        logits[:, self.cls] = 10.0
        return logits


class _VaryingIdm(torch.nn.Module):
    """Argmax actually follows the frame, so the labels are usable."""

    def __init__(self, num_actions: int = N_ACTIONS) -> None:
        super().__init__()
        # a real IdmConfig: save_idm() serialises it with as_dict()
        self.config = IdmConfig(image_size=16, embed_dim=16, num_actions=num_actions)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, frame_a, frame_b):
        key = (frame_a.mean(dim=(1, 2, 3)) * 255).long() % self.config.num_actions
        return torch.nn.functional.one_hot(key, self.config.num_actions).float() * 10.0


def _dataset(constant: bool, windows: int = 12, w_s: int = 4) -> dict:
    actions = np.zeros((windows, w_s, N_ACTIONS), dtype=np.float32)
    for i in range(windows):
        for j in range(w_s):
            cls = 3 if constant else (i * w_s + j) % N_ACTIONS
            actions[i, j, cls] = 1.0
    return {"actions": actions}


def test_stats_report_a_constant_labelling():
    stats = pseudo_label_stats(_dataset(True), N_ACTIONS)
    assert stats["majority_share"] == pytest.approx(1.0)
    assert stats["classes_used"] == 1
    assert stats["entropy"] == pytest.approx(0.0)


def test_stats_report_a_spread_labelling():
    stats = pseudo_label_stats(_dataset(False), N_ACTIONS)
    assert stats["majority_share"] < 0.5
    assert stats["classes_used"] > 1
    assert stats["entropy"] > 1.0


def test_stats_ignore_the_no_predecessor_pad_column():
    """Frame 0 of every window is all-zero padding, not a noop label.

    Counting it would inflate the majority share with padding and make a
    genuinely degenerate labelling look healthier than it is.
    """
    actions = np.zeros((5, 4, N_ACTIONS), dtype=np.float32)
    # only the pad column is class 0; every real label is class 7
    actions[:, 1:, 7] = 1.0
    stats = pseudo_label_stats({"actions": actions}, N_ACTIONS)
    assert stats["classes_used"] == 1
    assert stats["majority_share"] == pytest.approx(1.0)


def _run_bootstrap(idm, tmp_path):
    b = Bootstrapper(BootstrapConfig(
        image_size=16, w_s=4, w_l=2, idm_epochs=1, policy_epochs=1,
        batch_size=4, device="cpu",
    ))
    policy = AshPolicy(AshPolicyConfig(image_size=16, w_s=4, w_l=2, num_layers=1,
                                       num_actions=N_ACTIONS))
    rng = np.random.default_rng(0)

    def loader(ids=None):
        yield "v", rng.integers(0, 255, (20, 16, 16, 3), dtype=np.uint8)

    return b.run(policy, idm, _K(), [], loader, tmp_path)


def test_bootstrap_skips_policy_update_on_constant_labels(tmp_path):
    report = _run_bootstrap(_ConstantIdm(), tmp_path)

    assert report["policy"] == [], "pi must not be trained on a constant target"
    assert report["pseudo_labels"], "the degeneracy must be reported, not hidden"
    assert report["pseudo_labels"][0]["majority_share"] == pytest.approx(1.0)
    assert report["pseudo_labels"][0]["video"] == "v"


def test_bootstrap_updates_policy_when_labels_vary(tmp_path):
    report = _run_bootstrap(_VaryingIdm(), tmp_path)

    assert report["policy"], "usable labels must still train pi"
    assert report["pseudo_labels"][0]["majority_share"] <= 0.9


def test_logit_diagnosis_separates_bias_from_input():
    """A collapsed labelling must report *why* it collapsed.

    A per-class bias that no frame can outvote and a genuinely constant input
    both show majority_share 1.0; telling them apart needs the logits split into
    temporal movement and bias spread.  Measured on a real IDM: 0.008 against
    0.13, i.e. the argmax never depended on the frame.
    """
    from ash.loop.bootstrap import logit_diagnosis

    rng = np.random.default_rng(0)
    # Bias-dominated: a fixed per-class offset, almost no frame-to-frame motion.
    bias = np.tile(rng.normal(scale=0.13, size=16), (200, 1)).astype(np.float32)
    bias += rng.normal(scale=0.008, size=bias.shape).astype(np.float32)
    d = logit_diagnosis(bias)
    assert d["bias_over_temporal"] > 1.0, d
    assert 0.0 < d["logit_temporal_std"] < 0.05
    assert d["logit_bias_spread"] > 0.05

    # Input-driven: the classes move as much as they differ.
    driven = rng.normal(size=(200, 16)).astype(np.float32) * 2.0
    d2 = logit_diagnosis(driven)
    assert d2["bias_over_temporal"] < 1.0, d2

    assert logit_diagnosis(np.zeros((0, 16), dtype=np.float32)) == {}
    assert logit_diagnosis(None) == {}


def test_report_carries_the_logit_diagnosis(tmp_path):
    """It has to reach the report, not just be computable."""
    report = _run_bootstrap(_ConstantIdm(), tmp_path)
    stats = report["pseudo_labels"][0]
    assert "logit_temporal_std" in stats and "logit_bias_spread" in stats
    # _ConstantIdm returns a hard-coded vector: zero temporal movement.
    assert stats["logit_temporal_std"] == 0.0
