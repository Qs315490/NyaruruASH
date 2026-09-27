"""DINOv2 frame embeddings for key-moment discovery and retrieval.

Every observation is embedded once with DINOv2 (frozen, no finetuning) and the
resulting L2-normalized vector is used by two consumers:

- key-moment discovery: HDBSCAN clusters over these vectors (memory/kdm.py)
- retrieval: cosine similarity between trajectory and corpus frames (retrieval/)

Frames arrive as (H, W, C) uint8 arrays; batching keeps the ViT inference off
the per-step critical path.

AMD/ROCm note: run with HSA_OVERRIDE_GFX_VERSION=11.0.0 (gfx1101).  DINOv2 is
inference-only here, so the model and batch are kept small and moved to CPU
when VRAM is contended by the game process.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# The paper uses DINOv2 ViT-S/14; the small variant is plenty for 2D game
# frames where visual identity is largely color/layout, not texture.
_DINO_MODEL = "dinov2_vits14"
_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)


class FrameEmbedder:
    """Frozen DINOv2 that turns frames into L2-normalized embeddings."""

    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # weights_only / trust concerns do not apply: these are torch hub
        # checkpoints hosted by meta, no custom code.
        self.model: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", _DINO_MODEL, verbose=False
        ).to(self.device).eval()
        self._mean = torch.tensor(_DINO_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_DINO_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def embed(self, frames: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Embed an (N, H, W, C) uint8 array; returns (N, D) float32, L2-normalized."""
        if frames.ndim != 4:
            raise ValueError(f"frames must be (N,H,W,C), got {frames.shape}")
        out: list[np.ndarray] = []
        for i in range(0, len(frames), batch_size):
            batch = torch.from_numpy(frames[i : i + batch_size].copy())
            x = batch.to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - self._mean) / self._std
            # DINOv2 expects multiples of the patch size; it resizes internally
            # only via interpolate_pos_encoding, so pad to the next multiple.
            feats = self.model(x)
            feats = torch.nn.functional.normalize(feats, dim=1)
            out.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)
