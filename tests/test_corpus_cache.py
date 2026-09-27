"""The corpus embedding cache must be reused, and must not go stale silently.

Building the retrieval index costs one DINOv2 pass over the whole corpus - the
noisy 100%-GPU minute at the start of every run - so it is cached.  The cache
path defaulted to None, which meant the cache was written and never read: every
run re-embedded everything.

The other half is staleness.  A stale index does not fail: retrieval simply
scores against the wrong frames.  So the cache carries a fingerprint of the
corpus files and the embedding resolution, and a mismatch rebuilds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ash.cli.main import _corpus_embeddings, _corpus_fingerprint


class _CountingEmbedder:
    """Stands in for DINOv2; counts how many frames it was asked to embed."""

    def __init__(self, image_size: int = 256) -> None:
        self.image_size = image_size
        self.frames = 0
        self.calls = 0

    def embed(self, frames, batch_size=64):
        self.calls += 1
        self.frames += len(frames)
        return np.zeros((len(frames), 4), dtype=np.float32)


def _corpus(tmp_path, frames: int = 5):
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    np.savez_compressed(d / "v1.npz", frames=np.zeros((frames, 8, 8, 3), dtype=np.uint8))
    return d


def test_second_call_loads_the_cache_instead_of_re_embedding(tmp_path):
    d = _corpus(tmp_path)
    first = _CountingEmbedder()
    _corpus_embeddings(None, str(d), first)
    assert first.frames == 5, "the first call must embed"

    second = _CountingEmbedder()
    out = _corpus_embeddings(None, str(d), second)
    assert second.frames == 0, "the second call must read the cache, not re-embed"
    assert [name for name, _ in out] == ["v1"]


def test_cache_is_invalidated_when_the_corpus_changes(tmp_path):
    d = _corpus(tmp_path)
    _corpus_embeddings(None, str(d), _CountingEmbedder())

    np.savez_compressed(d / "v2.npz", frames=np.zeros((3, 8, 8, 3), dtype=np.uint8))

    again = _CountingEmbedder()
    out = _corpus_embeddings(None, str(d), again)
    assert again.frames == 8, "a changed corpus must be re-embedded"
    assert sorted(name for name, _ in out) == ["v1", "v2"]


def test_cache_is_invalidated_when_the_resolution_changes(tmp_path):
    d = _corpus(tmp_path)
    _corpus_embeddings(None, str(d), _CountingEmbedder(image_size=256))

    other = _CountingEmbedder(image_size=128)
    _corpus_embeddings(None, str(d), other)
    assert other.frames == 5, (
        "embeddings at another resolution are not interchangeable - DINOv2 "
        "features depend on the size they were computed at"
    )


def test_unversioned_cache_is_rebuilt(tmp_path):
    """A cache written before the fingerprint existed must not be trusted."""
    d = _corpus(tmp_path)
    stale = d.parent / (d.name + "-embeddings.npz")
    np.savez_compressed(stale, v1=np.zeros((5, 4), dtype=np.float32))

    emb = _CountingEmbedder()
    _corpus_embeddings(None, str(d), emb)
    assert emb.frames == 5, "no fingerprint means unknown provenance: rebuild"


def test_fingerprint_tracks_names_and_sizes(tmp_path):
    d = _corpus(tmp_path)
    a = _corpus_fingerprint(str(d), 256)
    assert json.loads(a)["image_size"] == 256
    np.savez_compressed(d / "v2.npz", frames=np.zeros((3, 8, 8, 3), dtype=np.uint8))
    assert _corpus_fingerprint(str(d), 256) != a


@pytest.mark.parametrize("size", [128, 256])
def test_fingerprint_carries_the_resolution(tmp_path, size):
    d = _corpus(tmp_path)
    assert json.loads(_corpus_fingerprint(str(d), size))["image_size"] == size
