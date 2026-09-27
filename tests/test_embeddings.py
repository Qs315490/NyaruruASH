"""DINOv2 input-shape handling.

DINOv2's patch embedding asserts H % patch == 0 and W % patch == 0 and does not
resize for you.  The project default frame size is 256 and the patch size is 14,
and 256 % 14 != 0 - so every embed() call died with

    AssertionError: Input image height 256 is not a multiple of patch height 14

This was caught by scripts/verify_retrieval.py against the real corpus, not by
the unit tests, because the old tests never called the model with the real frame
size.  These tests pin the shape contract without needing the 80 MB checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ash.memory.embeddings import DEFAULT_IMAGE_SIZE, FrameEmbedder  # noqa: E402


def test_patch_grid_rounds_up_to_multiple():
    x = torch.zeros(2, 3, 256, 256)
    out = FrameEmbedder._fit_patch_grid(x)
    assert out.shape[-2] % 14 == 0 and out.shape[-1] % 14 == 0
    # 256 -> 266 (19 * 14): round UP, never down, so no field of view is lost.
    assert out.shape[-2] == 266 and out.shape[-1] == 266


def test_patch_grid_is_identity_when_already_aligned():
    x = torch.rand(1, 3, 252, 252)
    assert FrameEmbedder._fit_patch_grid(x) is x


@pytest.mark.parametrize("size", [128, 224, 256, 300])
def test_common_frame_sizes_become_embeddable(size):
    """Whatever the caller passes, the tensor handed to the model is aligned."""
    out = FrameEmbedder._fit_patch_grid(torch.zeros(1, 3, size, size))
    assert out.shape[-2] % 14 == 0 and out.shape[-1] % 14 == 0


def test_patch_grid_preserves_batch_and_channels():
    out = FrameEmbedder._fit_patch_grid(torch.zeros(3, 3, 256, 256))
    assert out.shape[0] == 3 and out.shape[1] == 3


def test_canonicalize_maps_any_resolution_to_fixed_size():
    """Whatever the caller passes in, the model sees one resolution.

    The corpus is extracted at 256x256, but live game frames arrive at the
    window's native size.  Without this step the two are embedded at different
    scales, so K is fit on one patch grid and queried on another.
    """
    for h, w in [(624, 816), (720, 1280), (300, 400), (1080, 1920)]:
        out = FrameEmbedder._canonicalize(torch.zeros(2, 3, h, w), DEFAULT_IMAGE_SIZE)
        assert out.shape == (2, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)


def test_canonicalize_is_identity_when_already_canonical():
    x = torch.rand(1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    assert FrameEmbedder._canonicalize(x, DEFAULT_IMAGE_SIZE) is x


def test_game_frame_resolution_does_not_change_model_input():
    """The same picture at two resolutions must reach the model identically.

    Area-downscaling a nearest-integer-upscaled image recovers the original
    exactly, so this pins that the pipeline really is resolution-invariant.
    """
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    big = np.repeat(np.repeat(frame, 3, axis=0), 3, axis=1)  # 768x768

    def prep(arr):
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].float() / 255.0
        return FrameEmbedder._fit_patch_grid(
            FrameEmbedder._canonicalize(x, DEFAULT_IMAGE_SIZE)
        )

    a, b = prep(frame), prep(big)
    assert a.shape == b.shape == (1, 3, 266, 266)
    assert torch.allclose(a, b, atol=1e-4)


def test_resolutions_diverge_without_canonicalization():
    """Documents the failure the canonicalisation step exists to prevent."""
    corpus_like = FrameEmbedder._fit_patch_grid(torch.zeros(1, 3, 256, 256))
    game_like = FrameEmbedder._fit_patch_grid(torch.zeros(1, 3, 624, 816))
    assert corpus_like.shape != game_like.shape
