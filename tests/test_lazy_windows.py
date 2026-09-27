"""Corpus windows must not be materialized.

`build_policy_dataset` used to assemble them with np.stack, giving
(n_win, w_s, H, W, 3) float32.  For one 9405-frame corpus video at 128x128 with
w_s=32 that is 59 GB, and the memory bank added 15 GB more - on a 16 GB machine
the full-corpus bootstrap could never finish.  Measured: 18 minutes of one core
at 100% and not one artifact.  The small `corpus-live` subset had hidden it.

So the dataset stores the frame buffer once and the *index* of each window, and
these tests pin both halves of that: the memory bound, and that a batch is
bit-identical to the windows the old code would have built.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ash.actions.space import ActionSpace  # noqa: E402
from ash.loop.bootstrap import (  # noqa: E402
    BootstrapConfig,
    Bootstrapper,
    WindowDataset,
)

N_ACTIONS = len(ActionSpace.minimal())


class _FakeIdm(torch.nn.Module):
    """One class per frame pair, so the labels are predictable."""

    def __init__(self, num_actions: int = N_ACTIONS) -> None:
        super().__init__()
        from ash.models.idm import IdmConfig

        self.config = IdmConfig(image_size=32, embed_dim=16, num_actions=num_actions)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, frame_a, frame_b):
        return torch.zeros(frame_a.shape[0], self.config.num_actions)


class _K:
    """Fires on a fixed set of frames, so memories have something to reference."""

    def __init__(self, at) -> None:
        self.at = set(at)

    class _Emb:
        def embed(self, frames, batch_size=64):
            return np.zeros((len(frames), 4), dtype=np.float32)

    embedder = _Emb()

    def classify_sequence(self, embeddings):
        return np.array([i in self.at for i in range(len(embeddings))], dtype=bool)

    def fit(self, embeddings, trajectory_ids):
        return {"clusters_total": 0, "clusters_kept": 0, "noise_rate": 1.0}


def _dataset(T=200, image_size=32, w_s=16, w_l=4, key_at=(5, 40, 90)):
    b = Bootstrapper(BootstrapConfig(
        image_size=image_size, w_s=w_s, w_l=w_l, device="cpu",
    ))
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 255, (T, 48, 48, 3), dtype=np.uint8)
    return b.build_policy_dataset(obs, _FakeIdm(), _K(key_at)), b, obs


def test_dataset_stores_one_frame_buffer_not_one_copy_per_window():
    ds, _, obs = _dataset()
    assert isinstance(ds, WindowDataset)
    # One buffer of uint8 frames: the old code held float32 windows, i.e.
    # w_s=16 copies of every frame.
    assert ds.frames.shape == (200, 32, 32, 3)
    assert ds.frames.dtype == np.uint8
    assert ds.frames.nbytes == 200 * 32 * 32 * 3
    total = ds.frames.nbytes + ds.masks.nbytes + ds.mem_idx.nbytes
    assert total < 3 * ds.frames.nbytes, (
        "the dataset must not hold a second copy of the frames: %d bytes" % total
    )
    # A batch is what makes the windows; asking for the whole array is refused
    # rather than silently attempting tens of GB on a real corpus video.
    with pytest.raises(KeyError):
        ds["frames"]
    with pytest.raises(KeyError):
        ds["memories"]


def test_batch_is_identical_to_materialized_windows():
    """The lazy path must produce exactly what np.stack used to produce."""
    ds, _, _ = _dataset()
    idx = np.array([0, 3, 37, 180])
    got = ds.batch(idx)["frames"]
    want = np.stack([ds.frames[i : i + ds.w_s] for i in idx]).astype(np.float32) / 255.0
    assert np.array_equal(got, want)
    assert got.shape == (4, ds.w_s, ds.image_size, ds.image_size, 3)

    acts = ds["actions"]
    want_acts = np.stack([ds.masks[i : i + ds.w_s] for i in range(len(ds))])
    assert np.array_equal(acts, want_acts)


def test_memory_slots_point_at_key_moment_frames():
    ds, _, _ = _dataset(key_at=(5, 40, 90), w_l=2)
    batch = ds.batch(np.arange(len(ds)))["memories"]
    # Window 0 starts before any key moment: nothing to reference.
    assert not batch[0].any()
    # Window starting at 6 has frame 5 available; it lands in the LAST slot
    # (the prefix is left-aligned padding), which is the paper's ordering.
    win_at_6 = batch[6]
    assert not win_at_6[0].any()
    assert np.array_equal(win_at_6[1], ds.frames[5].astype(np.float32) / 255.0)
    # Window starting at 91 has both 5 and 40 before it; only the newest w_l stay.
    win_at_91 = batch[91]
    assert np.array_equal(win_at_91[0], ds.frames[40].astype(np.float32) / 255.0)
    assert np.array_equal(win_at_91[1], ds.frames[90].astype(np.float32) / 255.0)


def test_policy_still_trains_on_the_lazy_dataset(tmp_path):
    """The refactor must not have quietly disabled the policy update."""
    from ash.loop.bootstrap import BootstrapConfig as C
    from ash.loop.bootstrap import Bootstrapper as B
    from ash.models.ash_policy import AshPolicy, AshPolicyConfig

    b = B(C(image_size=32, w_s=16, w_l=4, idm_epochs=1, policy_epochs=1,
            batch_size=4, device="cpu"))
    ds, _, _ = _dataset()
    policy = AshPolicy(AshPolicyConfig(image_size=32, w_s=16, w_l=4, num_layers=1,
                                       num_actions=N_ACTIONS))
    report = b.update_policy(policy, ds)
    assert np.isfinite(report["policy_val"])


def test_corpus_embeddings_are_reused_not_recomputed(tmp_path):
    """A caller that already has the DINOv2 matrix must not cost a ViT pass.

    The retrieval index holds every corpus video's embeddings, yet both corpus
    passes in the bootstrap used to re-embed the video - twice per round - to
    recompute frozen, deterministic vectors that were already in memory.
    """
    from ash.loop.bootstrap import Bootstrapper as B, BootstrapConfig as C

    class _Exploding:
        """Any call means the embeddings were recomputed."""

        def embed(self, frames, batch_size=64):
            raise AssertionError("the corpus was re-embedded despite a cached index")

    rng = np.random.default_rng(0)
    obs = rng.integers(0, 255, (60, 16, 16, 3), dtype=np.uint8)
    embs = rng.normal(size=(60, 8)).astype(np.float32)
    kdm = _K(())
    kdm.embedder = _Exploding()

    b = B(C(image_size=16, w_s=8, w_l=2, idm_epochs=1, policy_epochs=1,
            batch_size=4, device="cpu"))
    ds = b.build_policy_dataset(obs, _FakeIdm(), kdm, embs)
    assert len(ds) == 60 - 8 + 1

    # And the full bootstrap: the K refit pass must reuse the index too.
    from ash.models.ash_policy import AshPolicy, AshPolicyConfig

    policy = AshPolicy(AshPolicyConfig(image_size=16, w_s=8, w_l=2, num_layers=1,
                                       num_actions=N_ACTIONS))

    def loader(ids=None):
        yield "v", obs

    report = b.run(policy, _FakeIdm(), kdm, [], loader, tmp_path,
                   retrieved_ids=["v"], corpus_embeddings={"v": embs})
    assert report["kdm"]["clusters_total"] == 0   # _K.fit is a stub
    assert "clusters_kept" in report["kdm"]
