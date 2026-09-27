"""The corpus loader must prefer the memory-mappable form.

Measured on this machine: materializing a 256x256 corpus video costs 2-2.6 GB of
anonymous memory the kernel cannot reclaim, which pushed the system into swap and
stalled the per-video bootstrap phase for minutes (three separate rounds).  The
uncompressed form is the same bytes, read as evictable page cache.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ash.cli.main import _corpus_loader


def _write(tmp_path, stem, frames):
    np.savez(tmp_path / f"{stem}.npz", frames=frames)


def test_loader_prefers_the_mmap_form(tmp_path):
    frames = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
    np.save(tmp_path / "video.npy", frames)
    _write(tmp_path, "video", frames)
    loaded = dict(_corpus_loader(str(tmp_path))())
    assert isinstance(loaded["video"], np.memmap), (
        "the packed corpus must be memory-mapped, not decompressed into RAM"
    )
    assert np.array_equal(loaded["video"], frames)


def test_loader_falls_back_to_the_npz(tmp_path):
    frames = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
    _write(tmp_path, "video", frames)
    loaded = dict(_corpus_loader(str(tmp_path))())
    assert np.array_equal(loaded["video"], frames)


def test_loader_restricts_to_the_retrieved_ids(tmp_path):
    for stem in ("a", "b"):
        _write(tmp_path, stem, np.zeros((2, 4, 4, 3), dtype=np.uint8))
    assert [k for k, _ in _corpus_loader(str(tmp_path))(ids=["b"])] == ["b"]


def test_recorded_sessions_join_the_corpus(tmp_path):
    """K is fit on D^I, so a recording of the rooms the videos skip belongs in it.

    Measured before this existed: in every round the agent sat at cosine 0.77-0.92
    to its nearest corpus frame, no corpus frame came within 0.95 of any live
    frame, and `key_moments` was 0 - the corpus is speedruns and nobody films the
    starting house or an NPC room.
    """
    corpus = tmp_path / "corpus"
    recs = tmp_path / "recordings"
    corpus.mkdir()
    recs.mkdir()
    video = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    session = np.full((2, 8, 8, 3), 7, dtype=np.uint8)
    _write(corpus, "video", video)
    # A recorded session spells the array `observations`, not `frames`.
    np.savez(recs / "human-001.npz", observations=session,
             control_masks=np.zeros(2, dtype=np.int64))
    loaded = dict(_corpus_loader(str(corpus), str(recs))())
    assert set(loaded) == {"video", "human-001"}
    assert np.array_equal(loaded["human-001"], session)
    assert np.array_equal(loaded["video"], video)


def test_a_missing_recordings_dir_is_ignored(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "video", np.zeros((2, 4, 4, 3), dtype=np.uint8))
    got = dict(_corpus_loader(str(corpus), str(tmp_path / "nope"))())
    assert set(got) == {"video"}


def test_the_fingerprint_sees_packed_files_and_recordings(tmp_path):
    """A fingerprint that only globs *.npz went blind when the corpus was packed.

    The originals were deleted and replaced by `.npy`, so the file list became
    empty and any later corpus change would have been silently ignored - a stale
    index reused, which is the failure the fingerprint exists to prevent.
    """
    from ash.cli.main import _corpus_fingerprint

    corpus = tmp_path / "corpus"
    recs = tmp_path / "recordings"
    corpus.mkdir()
    recs.mkdir()
    np.save(corpus / "video.npy", np.zeros((2, 4, 4, 3), dtype=np.uint8))
    before = _corpus_fingerprint(str(corpus), 128, str(recs))
    assert "video.npy" in before, "packed corpus files must be part of the identity"
    empty = json.dumps({"v": 1, "image_size": 128, "files": []}, sort_keys=True)
    assert before != empty, "an empty fingerprint is what hid the packed corpus"

    np.save(recs / "human-001.npy", np.zeros((2, 4, 4, 3), dtype=np.uint8))
    after = _corpus_fingerprint(str(corpus), 128, str(recs))
    assert after != before, "adding a recording must invalidate the cached index"
