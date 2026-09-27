"""Training loop for the inverse dynamics model.

Pairs are (frame[t], frame[t+1]) -> mask[t+1]: the mask recorded for a frame is
the key state at that frame, so the model is asked "which keys explain how the
picture got from t to t+1".  Pairs never straddle an episode boundary - the
frames on either side of a cut have no causal relation.

The dataset npz keeps every observation of the corpus in one flat array.  It is
loaded once (646 MB for 24447 frames at 128x128) and indexed lazily, which is
both faster and simpler than streaming the videos again at train time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ash.models.idm import IdmConfig, IdmModel, save_idm
from ash.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class IdmTrainConfig:
    data_path: str = "data/idm-human.npz"
    out_path: str = "models/idm-human.pt"
    epochs: int = 6
    batch_size: int = 64
    lr: float = 3e-4
    val_fraction: float = 0.08
    num_workers: int = 2
    seed: int = 1
    # 0 = infer from the dataset.  The trunk is built for a fixed resolution,
    # so a mismatch between the packed frames and this value fails deep inside
    # the first linear layer; inferring it removes that trap.
    image_size: int = 0
    # ImpalaCNN width preset.  VPT names these 1x/2x/3x; the trunk channels are
    # 16/32/32 scaled by 4/8/16, so 2x roughly triples the parameters.  The
    # 12 GB card has room for 2x at 256px where the old 4 GB card did not.
    width: str = "1x"
    embed_dim: int = 256
    # Mixed precision.  On the RX 7700 XT (RDNA3 matrix cores) fp16 is both
    # faster and leaner than fp32 - measured in docs/gpu-benchmarks.md - which
    # is what makes 2x at 256px fit in 12 GB.  Off by default so the flag is
    # explicit, exactly as BcConfig.amp is.
    amp: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class PairDataset(Dataset):
    """(frame[t], frame[t+1]) -> mask[t+1], weight[t+1] pairs of one corpus."""

    def __init__(self, npz_path: str | Path, indices: np.ndarray, num_keys: int = 17) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.obs = data["observations"]
        self.num_keys = num_keys
        # control_masks is a packed 17-bit integer per frame; the head needs one
        # float target per key, so bits are unpacked once up front.
        masks = data["control_masks"].astype(np.int64)
        self.masks = np.stack(
            [(masks >> k) & 1 for k in range(num_keys)], axis=1
        ).astype(np.float32)
        self.weights = data["weights"].astype(np.float32)
        self.episode_ids = data["episode_ids"]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        t = int(self.indices[i])
        a = torch.from_numpy(np.ascontiguousarray(self.obs[t]))
        b = torch.from_numpy(np.ascontiguousarray(self.obs[t + 1]))
        mask = torch.from_numpy(np.ascontiguousarray(self.masks[t + 1]))
        weight = torch.tensor(float(self.weights[t + 1]))
        return a, b, mask, weight


def build_pair_indices(episode_ids: np.ndarray, val_fraction: float, seed: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Valid pair positions (t, t+1 within one episode), split randomly.

    A per-episode split was tried first and rejected: with 13 recordings whose
    key coverage varies wildly (one episode presses only X, another only ESC), an
    8% episode hold-out has almost none of the rare keys, so its metrics say
    nothing about the keys that matter.  Pairs are instead drawn at random; the
    leakage this permits (near-duplicate frames across the split) is acceptable
    because the IDM is used as an annotator, not as a generalising policy.
    """
    rng = np.random.default_rng(seed)
    boundaries = np.flatnonzero(np.diff(episode_ids)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(episode_ids)]])
    pairs = []
    for s, e in zip(starts, ends, strict=True):
        pairs.append(np.arange(s, e - 1))
    pairs = np.concatenate(pairs)
    shuffled = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_fraction))
    val = np.sort(shuffled[:n_val])
    train = np.sort(shuffled[n_val:])
    return pairs[train], pairs[val]


def train_idm(config: IdmTrainConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(config.data_path, allow_pickle=True)
    episode_ids = data["episode_ids"]
    names = [str(n) for n in data["control_names"]]
    train_idx, val_idx = build_pair_indices(episode_ids, config.val_fraction, config.seed)
    log.info("pairs: %d train / %d val (episodes held out: %d)",
             len(train_idx), len(val_idx), len(np.unique(episode_ids[val_idx])))

    train_ds = PairDataset(config.data_path, train_idx)
    val_ds = PairDataset(config.data_path, val_idx)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, num_workers=config.num_workers,
        pin_memory=True,
    )

    image_size = config.image_size or int(data["observations"].shape[1])
    model = IdmModel(
        IdmConfig(
            num_keys=len(names),
            image_size=image_size,
            width=config.width,
            embed_dim=config.embed_dim,
        )
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "idm: image_size=%d keys=%d width=%s params=%.2fM",
        image_size, len(names), config.width, n_params / 1e6,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * len(train_loader)
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def evaluate() -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        hits = 0
        count = 0
        with torch.no_grad():
            for a, b, mask, weight in val_loader:
                # The model documents (B, H, W, C) input and adds its own time
                # axis; permuting to (B, C, H, W) here double-transposed the
                # tensor and the trunk then saw the wrong channel count.
                a = a.float().to(device) / 255.0
                b = b.float().to(device) / 255.0
                mask = mask.to(device)
                weight = weight.to(device)
                # autocast the trunk only: the loss stays fp32 so the BCE
                # targets keep full precision, as the trainer always did.
                with torch.autocast("cuda", enabled=use_amp):
                    logits = model(a, b)
                losses = loss_fn(logits.float(), mask).mean(dim=1)
                total_loss += float((losses * weight).sum())
                preds = (logits.float().sigmoid() > 0.5).float()
                hits += int(((preds == mask).all(dim=1) * weight).sum())
                count += int(weight.sum())
        return total_loss / max(1, count), hits / max(1, count)

    history = []
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        seen = 0
        for a, b, mask, weight in train_loader:
            a = a.float().to(device) / 255.0
            b = b.float().to(device) / 255.0
            mask = mask.to(device)
            weight = weight.to(device)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(a, b)
            # fp32 loss, fp32 reduced precision path for the gradient scale
            losses = loss_fn(logits.float(), mask).mean(dim=1)
            loss = (losses * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * len(a)
            seen += len(a)
        val_loss, val_exact = evaluate()
        history.append({
            "epoch": epoch + 1,
            "train_loss": running / max(1, seen),
            "val_loss": val_loss,
            "val_exact_match": val_exact,
        })
        log.info("epoch %d/%d  train %.4f  val %.4f  val exact %.3f",
                 epoch + 1, config.epochs, running / max(1, seen), val_loss, val_exact)

    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_idm(model, out_path)
    report = {
        "config": config.as_dict(),
        "control_names": names,
        "history": history,
        "final_val_exact_match": history[-1]["val_exact_match"] if history else None,
    }
    (out_path.parent / (out_path.stem + "-report.json")).write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    log.info("saved %s", out_path)
    return report
