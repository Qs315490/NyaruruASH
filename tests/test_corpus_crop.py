"""The crop that removes the stream overlay from scraped corpus frames.

Every assertion here pins something that would otherwise fail silently: a crop
that reads the wrong axes (the corpus is (T, H, W, C) - slicing the first two
crops TIME, which produced "217 frames of a 9405-frame video" and a meaningless
similarity during this measurement), a wrapper that materializes the video it was
meant to keep mappable, or a cache fingerprint that misses a crop change and
reuses embeddings of the uncropped frames.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ash.data.corpus_crop import LIVE_ASPECT, CroppedFrames, crop_block, load_crops


def test_the_crop_takes_the_rect_and_restores_the_live_aspect():
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    out = crop_block(frame, (2, 1, 8, 4))          # x[2,8] y[1,4] -> (3, 6, 3)

    assert out.shape == (3, int(round(3 * LIVE_ASPECT)), 3)
    # The first and last kept columns are the rect's own, so the rect is what was
    # cut - not the first two axes.
    assert np.array_equal(out[..., 0, :], frame[1:4, 2, :])
    assert np.array_equal(out[..., -1, :], frame[1:4, 7, :])


def test_the_crop_does_not_read_the_time_axis():
    """(T, H, W, C): the spatial axes are the last three."""
    frames = np.arange(5 * 6 * 8 * 3, dtype=np.uint8).reshape(5, 6, 8, 3)
    out = crop_block(frames, (2, 1, 8, 4))
    assert out.shape[0] == 5, out.shape
    assert out.shape[1] == 3


def test_the_wrapper_keeps_the_frames_lazy():
    """It exists so the mmap stays an mmap: cropping up front is 1.6 GB a video."""

    class Counting:
        def __init__(self) -> None:
            self.shape = (10, 6, 8, 3)
            self.ndim = 4
            self.dtype = np.dtype(np.uint8)
            self.reads = 0

        def __len__(self) -> int:
            return 10

        def __getitem__(self, index):
            self.reads += 1
            return np.zeros((6, 8, 3), dtype=np.uint8)

    frames = Counting()
    wrapped = CroppedFrames(frames, (2, 1, 8, 4))
    assert frames.reads == 0, "constructing the wrapper must not read frames"
    assert wrapped.shape == (10, 3, 4, 3)
    assert len(wrapped) == 10
    wrapped[3]
    assert frames.reads == 1, "one access reads one frame, not the video"


def test_a_missing_or_broken_crop_file_means_no_crop(tmp_path):
    """A corpus without crop info is what the project ran on before."""
    assert load_crops(None) == {}
    assert load_crops(tmp_path / "absent.json") == {}

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert load_crops(broken) == {}

    for bad_rect in ([1, 2], ["a", "b", "c", "d"], [5, 0, 1, 4]):   # x1 <= x0
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"rects": {"v": bad_rect}}))
        assert load_crops(path) == {}, bad_rect


def test_the_cache_fingerprint_notices_a_crop_change(tmp_path):
    """Otherwise a crop edit silently reuses embeddings of the uncropped frames."""
    from ash.cli.main import _corpus_fingerprint

    corpus, crops = tmp_path / "corpus", tmp_path / "crops.json"
    corpus.mkdir()
    np.save(corpus / "v.npy", np.zeros((4, 6, 8, 3), dtype=np.uint8))
    crops.write_text(json.dumps({"rects": {"v": [0, 0, 8, 6]}}))

    before = _corpus_fingerprint(str(corpus), 256, None, str(crops))
    crops.write_text(json.dumps({"rects": {"v": [2, 1, 8, 6]}}))
    after = _corpus_fingerprint(str(corpus), 256, None, str(crops))

    assert before != after
    assert _corpus_fingerprint(str(corpus), 256, None, str(crops)) == after


def test_the_loader_hands_out_cropped_frames(tmp_path):
    from ash.cli.main import _corpus_loader

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    np.save(corpus / "v.npy", np.arange(4 * 6 * 8 * 3, dtype=np.uint8).reshape(4, 6, 8, 3))
    crops = tmp_path / "crops.json"
    crops.write_text(json.dumps({"rects": {"v": [2, 1, 8, 4]}}))

    loaded = dict(_corpus_loader(str(corpus), None, str(crops))())

    assert loaded["v"].shape == (4, 3, 4, 3)
    # Without the file the loader behaves exactly as before.
    plain = dict(_corpus_loader(str(corpus), None, None)())
    assert plain["v"].shape == (4, 6, 8, 3)


def test_an_unknown_stem_is_left_alone(tmp_path):
    """A recording has no overlay, and the crop file has no rect for it."""
    from ash.cli.main import _corpus_loader

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    np.save(corpus / "rec.npy", np.zeros((3, 6, 8, 3), dtype=np.uint8))
    crops = tmp_path / "crops.json"
    crops.write_text(json.dumps({"rects": {"other": [0, 0, 4, 4]}}))

    loaded = dict(_corpus_loader(str(corpus), None, str(crops))())
    assert loaded["rec"].shape == (3, 6, 8, 3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
