"""A run that dies in the bootstrap must still leave its round on disk.

The report used to be written once, after the whole loop returned.  A run that
was killed - or that crashed - during the bootstrap therefore erased the only
record of the round that had actually been played, including whether the agent
had seen a single key moment.  Measured: an 18-minute full-corpus bootstrap
died and left an empty output directory.

So the round is written as soon as its inference statistics and its D^R are
known, marked `bootstrap_pending` until the bootstrap fills it in.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ash.actions.space import ActionSpace  # noqa: E402
from ash.loop.orchestrator import LoopConfig, Orchestrator  # noqa: E402
from ash.models.ash_policy import AshPolicy, AshPolicyConfig  # noqa: E402
from ash.models.idm import IdmConfig, IdmModel  # noqa: E402

N_ACTIONS = len(ActionSpace.minimal())
H = W = 16


class _Embedder:
    def embed(self, frames, batch_size=64):
        return np.zeros((len(frames), 8), dtype=np.float32)


def _orchestrator(tmp_path, bootstrap_fn):
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 255, (6, H, W, 3), dtype=np.uint8)

    def runner(**kw):
        return {
            "trajectories": [{"obs": obs, "act": rng.integers(0, N_ACTIONS, 5)}],
            "random_trajectories": [],
            "memories": [list(range(2))],       # two key moments
            "key_stats": [{"evals": 5, "noise": 3, "in_key_cluster": 2, "fired": 2}],
            "steps": [6],
            "stuck": True,
            "aborted": None,
        }

    policy = AshPolicy(AshPolicyConfig(image_size=H, w_s=4, w_l=2, num_layers=1,
                                       num_actions=N_ACTIONS))
    idm = IdmModel(IdmConfig(image_size=H, embed_dim=16, num_actions=N_ACTIONS))
    return Orchestrator(
        policy, idm, None, _Embedder(),
        [("corpus-video", np.ones((10, 8), dtype=np.float32))],
        config=LoopConfig(delta=4, max_bootstraps=1, out_dir=tmp_path),
        runner=runner, bootstrap_fn=bootstrap_fn,
    )


def test_round_survives_a_bootstrap_that_raises(tmp_path):
    def boom(**kw):
        raise MemoryError("pretend the bootstrap ran out of memory")

    orch = _orchestrator(tmp_path, boom)
    with pytest.raises(MemoryError):
        orch.run()

    doc = json.loads((tmp_path / "loop-report.json").read_text())
    assert len(doc["rounds"]) == 1, "the played round must not be erased"
    entry = doc["rounds"][0]
    assert entry["stats"]["key_moments"] == [2]
    assert entry["stats"]["stuck"] is True
    assert entry["round"] == 0
    assert entry["bootstrap_pending"] is True
    assert "kdm" not in entry
    assert doc["done"] is False


def test_completed_round_has_no_pending_marker(tmp_path):
    def ok(**kw):
        return {"kdm": {"clusters_kept": 3}, "idm": {}, "policy": [], "pseudo_labels": []}

    doc = _orchestrator(tmp_path, ok).run()
    assert doc["done"] is True
    assert doc["bootstraps"] == 1
    entry = doc["rounds"][0]
    assert "bootstrap_pending" not in entry
    assert entry["kdm"] == {"clusters_kept": 3}
    assert entry["retrieved"] == ["corpus-video"]
    on_disk = json.loads((tmp_path / "loop-report.json").read_text())
    assert on_disk == doc


def test_round_stats_carry_key_verdicts_and_corpus_similarity(tmp_path):
    """A round with no key moment must say which kind of nothing it was."""
    def ok(**kw):
        return {"kdm": {}, "idm": {}, "policy": [], "pseudo_labels": []}

    doc = _orchestrator(tmp_path, ok).run()
    stats = doc["rounds"][0]["stats"]
    assert stats["key_stats"] == [
        {"evals": 5, "noise": 3, "in_key_cluster": 2, "fired": 2}
    ], "the runner's K verdicts must survive into the report"
    # The fake runner's frames are one repeated frame; the corpus index is a
    # single ones-matrix, so the best cosine is finite and <= 1.
    assert stats["best_corpus_cosine"] is not None
    assert -1.0 <= stats["best_corpus_cosine"] <= 1.0


def test_round_frames_are_dumped_next_to_the_report(tmp_path):
    """A round's frames must outlive the round.

    Every attempt to explain "K saw nothing" after the fact failed for want of
    them: the embeddings are not kept, and capturing later gives a frozen frame
    because the game is paused during the bootstrap.
    """
    def ok(**kw):
        return {"kdm": {}, "idm": {}, "policy": [], "pseudo_labels": []}

    doc = _orchestrator(tmp_path, ok).run()
    path = tmp_path / "round-000-frames.npz"
    assert path.exists()
    with np.load(path) as d:
        frames = d["frames"]
    assert frames.dtype == np.uint8
    assert frames.ndim == 4 and frames.shape[1:] == (H, W, 3)
    assert 0 < len(frames) <= 64
    assert doc["rounds"][0]["stats"]["key_stats"] is not None


def test_frame_dump_is_bounded_and_spread(tmp_path):
    """It samples across the round rather than taking the first 64 frames."""
    from ash.loop.orchestrator import Orchestrator

    long_obs = np.arange(400, dtype=np.uint8).reshape(400, 1, 1, 1) * np.ones((1, H, W, 3), np.uint8)
    got = Orchestrator._sample_frames({"obs": long_obs})
    assert len(got) == 64
    assert got[1, 0, 0, 0] > got[0, 0, 0, 0], "frames must be spread, not the head"

    short_obs = np.zeros((5, H, W, 3), dtype=np.uint8)
    assert len(Orchestrator._sample_frames({"obs": short_obs})) == 5
    assert len(Orchestrator._sample_frames({"obs": np.zeros((0, H, W, 3), np.uint8)})) == 0
