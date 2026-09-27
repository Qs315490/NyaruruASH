"""DINOv2 frame embeddings for key-moment discovery and retrieval.

Every observation is embedded once with DINOv2 (frozen, no finetuning) and the
resulting L2-normalized vector is used by two consumers:

- key-moment discovery: HDBSCAN clusters over these vectors (memory/kdm.py)
- retrieval: cosine similarity between trajectory and corpus frames (retrieval/)

Frames arrive as (H, W, C) uint8 arrays; batching keeps the ViT inference off
the per-step critical path.

AMD/ROCm note: do NOT set HSA_OVERRIDE_GFX_VERSION.  The installed wheel is
built for gfx1101 (RX 7700 XT) and native detection resolves to gfx1101;
forcing 11.0.0 makes the runtime claim gfx1100 and every kernel launch fails
with hipErrorInvalidKernelFile.  DINOv2 is inference-only here, so the model
and batch are kept small and moved to CPU when VRAM is contended by the game.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ash.utils.device import resolve_device

# The paper uses DINOv2 ViT-S/14; the small variant is plenty for 2D game
# frames where visual identity is largely color/layout, not texture.
_DINO_MODEL = "dinov2_vits14"
_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)


#: Resolution every frame is embedded at, and the size the corpus is extracted
#: at (scripts/build_corpus.py --image-size).  The two must agree: DINOv2
#: features depend on the resolution they are computed at, so a corpus embedded
#: at 256 and live game frames embedded at, say, 816x624 land in different parts
#: of feature space and K is fit on one scale while being queried at another.
DEFAULT_IMAGE_SIZE = 256


class FrameEmbedder:
    """Frozen DINOv2 that turns frames into L2-normalized embeddings."""

    def __init__(self, device: str | None = None, image_size: int = DEFAULT_IMAGE_SIZE) -> None:
        self.device = resolve_device(device)
        self.image_size = image_size
        # weights_only / trust concerns do not apply: these are torch hub
        # checkpoints hosted by meta, no custom code.
        self.model: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", _DINO_MODEL, verbose=False
        ).to(self.device).eval()
        self._mean = torch.tensor(_DINO_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_DINO_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def embed(self, frames: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Embed an (N, H, W, C) uint8 array; returns (N, D) float32, L2-normalized.

        Any input resolution is first resized to `image_size`, so a caller can
        pass raw game frames or pre-resized corpus frames interchangeably.
        """
        if frames.ndim != 4:
            raise ValueError(f"frames must be (N,H,W,C), got {frames.shape}")
        out: list[np.ndarray] = []
        for i in range(0, len(frames), batch_size):
            batch = torch.from_numpy(frames[i : i + batch_size].copy())
            x = batch.to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
            x = self._canonicalize(x, self.image_size)
            x = (x - self._mean) / self._std
            x = self._fit_patch_grid(x)
            feats = self.model(x)
            feats = torch.nn.functional.normalize(feats, dim=1)
            out.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    @staticmethod
    def _canonicalize(x: "torch.Tensor", image_size: int) -> "torch.Tensor":
        """Resize (N, C, H, W) to a fixed square size, area-interpolated.

        Area (average) interpolation matches the cv2.INTER_AREA the corpus
        extractor uses, so downscaling a game frame and downscaling in ffmpeg
        produce the same picture.  Already-canonical input is returned as-is.
        """
        if x.shape[-2] == image_size and x.shape[-1] == image_size:
            return x
        return torch.nn.functional.interpolate(
            x, size=(image_size, image_size), mode="area"
        )

    @staticmethod
    def _fit_patch_grid(x: "torch.Tensor", patch: int = 14) -> "torch.Tensor":
        """Resize so H and W are multiples of the ViT patch size.

        DINOv2's patch embedding asserts H % patch == 0 and W % patch == 0; it
        does NOT resize for you.  A 256x256 frame (the project default) is not a
        multiple of 14, so every embed() call used to die here with
        "Input image height 256 is not a multiple of patch height 14".  Round
        UP to the next multiple and interpolate: downscaling to 252 would throw
        away 4 pixels of a small frame, and rounding up keeps the full field of
        view.  The resize is bilinear on an already-normalized float tensor.
        """
        _, _, h, w = x.shape
        if h % patch == 0 and w % patch == 0:
            return x
        th = (h + patch - 1) // patch * patch
        tw = (w + patch - 1) // patch * patch
        return torch.nn.functional.interpolate(
            x, size=(th, tw), mode="bilinear", align_corners=False
        )
