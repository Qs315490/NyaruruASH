"""Behaviour cloning on recorded episodes.

This is the first half of the VPT recipe and, on a 4 GB GPU, the half that
actually produces a usable agent.  The second half - search over a rollback-able
environment, distilled back into the policy - lives in vpt.planner and only works
when the backend reports an exact rollback.

Design notes that matter on a small machine:

  * data is read from a packed .npz with mmap, so video decoding never happens
    during training;
  * frame stacking replaces the recurrent core (a 4-frame stack at 128x128 is a
    few MB per batch instead of a 1024-wide LSTM/transformer state);
  * the value head regresses the milestone potential rather than an MC return,
    which is dense, cheap and directly aligned with finishing the run;
  * mixed precision and gradient accumulation are switches, not assumptions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ash.models.policy import PolicyConfig, VptPolicy
from ash.utils.logging import get_logger
from ash.utils.seed import seed_everything

log = get_logger(__name__)


@dataclass
class BcConfig:
    data: list[str] = field(default_factory=list)
    out_dir: str = "runs/bc"
    epochs: int = 5
    batch_size: int = 32
    stack: int = 4
    image_size: int = 0
    lr: float = 3e-4
    weight_decay: float = 0.0
    milestone_weight: float = 0.1
    value_weight: float = 0.1
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    device: str = "auto"
    # Mixed precision is off by default, and that is a measurement, not caution:
    # on the GTX 1650 (sm_75) this build's fp16 path runs at 0.34 TFLOP/s against
    # 4.12 TFLOP/s for fp32, so --amp makes training an order of magnitude slower.
    # It stays available for hardware where fp16 actually pays off.
    amp: bool = False
    accum_steps: int = 1
    seed: int = 0
    width: str = "1x"
    head_hidden: int = 512
    log_every: int = 50
    max_epoch_steps: int = 0
    class_weighting: bool = True
    class_weight_power: float = 0.5

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class PackedDataset:
    """Memory-mapped (observations, actions, states, milestones) bundle."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError("no packed dataset at %s" % self.path)
        self._data = np.load(self.path, mmap_mode="r", allow_pickle=False)
        required = {"observations", "actions", "states", "episode_ids"}
        missing = required - set(self._data.files)
        if missing:
            raise ValueError("packed dataset is missing %s" % sorted(missing))
        self.observations = self._data["observations"]
        self.actions = self._data["actions"]
        self.states = self._data["states"]
        self.episode_ids = self._data["episode_ids"]
        if "milestones" in self._data.files:
            self.milestones = self._data["milestones"]
        else:
            self.milestones = np.zeros(len(self.actions), dtype=np.int64)
        if "action_masks" in self._data.files:
            self.action_masks = [int(v) for v in self._data["action_masks"].tolist()]
        else:
            self.action_masks = []
        self.num_milestones = int(self.milestones.max()) + 1 if len(self.milestones) else 0
        self._episode_slices = self._index_episodes()

    def _index_episodes(self) -> dict[int, tuple[int, int]]:
        slices: dict[int, tuple[int, int]] = {}
        ids = np.asarray(self.episode_ids)
        if len(ids) == 0:
            return slices
        boundaries = np.flatnonzero(np.diff(ids)) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(ids)]])
        for start, end in zip(starts, ends, strict=True):
            slices[int(ids[start])] = (int(start), int(end))
        return slices

    @property
    def episode_ids_sorted(self) -> list[int]:
        return sorted(self._episode_slices)

    def episode_length(self, episode: int) -> int:
        start, end = self._episode_slices[episode]
        return end - start

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.observations.shape[1:])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1]) if self.states.ndim == 2 else 0

    @property
    def num_actions(self) -> int:
        if self.action_masks:
            return len(self.action_masks)
        return int(self.actions.max()) + 1

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def sample_batch(
        self,
        batch_size: int,
        *,
        stack: int,
        episodes: Iterable[int] | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample anchor frames and reconstruct their frame stacks.

        A stack is clipped at the episode start rather than wrapped, so a batch
        never splices the end of one run onto the beginning of another.
        """
        rng = rng or np.random.default_rng()
        allowed = list(episodes) if episodes is not None else self.episode_ids_sorted
        if not allowed:
            raise ValueError("sample_batch called with no episodes")
        indices = np.empty(batch_size, dtype=np.int64)
        for b in range(batch_size):
            ep = int(rng.choice(allowed))
            start, end = self._episode_slices[ep]
            indices[b] = int(rng.integers(start, end))
        obs = np.empty((batch_size, stack) + self.observation_shape, dtype=np.uint8)
        states = np.empty((batch_size, stack, self.state_dim), dtype=np.float32)
        for b, t in enumerate(indices):
            ep = int(self.episode_ids[t])
            start, _ = self._episode_slices[ep]
            for s in range(stack):
                src = max(start, int(t) - (stack - 1 - s))
                obs[b, s] = self.observations[src]
                if self.state_dim:
                    states[b, s] = self.states[src]
        return {
            "images": obs,
            "states": states,
            "actions": np.asarray(self.actions[indices], dtype=np.int64),
            "milestones": np.asarray(self.milestones[indices], dtype=np.int64),
            "indices": indices,
        }


def train_bc(config: BcConfig) -> Path:
    """Train a policy by imitation and return the path of the best checkpoint."""
    if not config.data:
        raise ValueError("BcConfig.data must list at least one packed dataset")
    seed_everything(config.seed)
    device = config.resolved_device()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PackedDataset(config.data[0])
    if len(config.data) > 1:
        log.warning(
            "multiple datasets given, only %s is used; merge them into one .npz first",
            config.data[0],
        )
    rng = np.random.default_rng(config.seed)

    image_size = config.image_size or dataset.observation_shape[0]
    policy_config = PolicyConfig(
        num_actions=dataset.num_actions,
        state_dim=dataset.state_dim,
        num_milestones=dataset.num_milestones,
        width=config.width,
        image_size=image_size,
        head_hidden=config.head_hidden,
    )
    policy = VptPolicy(policy_config).to(device)
    log.info(
        "policy %s: %.2fM params | actions=%d state_dim=%d milestones=%d device=%s",
        config.width,
        policy.num_parameters() / 1e6,
        dataset.num_actions,
        dataset.state_dim,
        dataset.num_milestones,
        device,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.amp and device.type == "cuda"))
    use_amp = bool(config.amp and device.type == "cuda")

    episode_pool = dataset.episode_ids_sorted
    rng.shuffle(episode_pool)
    val_count = (
        max(1, int(len(episode_pool) * config.val_fraction)) if len(episode_pool) > 1 else 0
    )
    val_episodes = episode_pool[:val_count]
    train_episodes = episode_pool[val_count:] or episode_pool

    # Class weights are computed from the training split only: deriving them from
    # the whole dataset would leak the held-out episodes' action distribution.
    action_weights = _class_weights(dataset, train_episodes, config, device)
    if action_weights is not None:
        log.info(
            "class weights (power=%.2f): %s",
            config.class_weight_power,
            [round(float(w), 2) for w in action_weights.tolist()],
        )

    train_steps = sum(dataset.episode_length(ep) for ep in train_episodes)
    batches_per_epoch = max(1, train_steps // config.batch_size)
    if config.max_epoch_steps:
        batches_per_epoch = min(batches_per_epoch, config.max_epoch_steps)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = out_dir / "last.ckpt"

    for epoch in range(config.epochs):
        policy.train()
        epoch_start = time.time()
        running = {"loss": 0.0, "action": 0.0, "milestone": 0.0, "value": 0.0, "accuracy": 0.0}
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index in range(batches_per_epoch):
            batch = dataset.sample_batch(
                config.batch_size, stack=config.stack, episodes=train_episodes, rng=rng
            )
            images = torch.from_numpy(batch["images"]).to(device)
            states = torch.from_numpy(batch["states"]).to(device)
            actions = torch.from_numpy(batch["actions"]).to(device)
            milestones = torch.from_numpy(batch["milestones"]).to(device)
            with torch.autocast("cuda", enabled=use_amp):
                out = policy(images, states if dataset.state_dim else None)
                logits = out["action_logits"][:, -1]
                action_loss = F.cross_entropy(logits, actions, weight=action_weights)
                loss = action_loss
                milestone_loss = torch.zeros((), device=device)
                if policy.milestone_head is not None and dataset.num_milestones > 0:
                    milestone_loss = F.cross_entropy(
                        out["milestone_logits"][:, -1],
                        milestones.clamp(0, dataset.num_milestones - 1),
                    )
                    loss = loss + config.milestone_weight * milestone_loss
                value_loss = torch.zeros((), device=device)
                if policy.config.num_milestones > 1:
                    progress = milestones.clamp(0, dataset.num_milestones - 1).float()
                    target = (dataset.num_milestones - 1 - progress) / (dataset.num_milestones - 1)
                    value_loss = F.mse_loss(out["value"][:, -1], target)
                    loss = loss + config.value_weight * value_loss
            scaled = loss / config.accum_steps
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            if (batch_index + 1) % config.accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running["loss"] += float(loss.item())
            running["action"] += float(action_loss.item())
            running["milestone"] += float(milestone_loss.item())
            running["value"] += float(value_loss.item())
            running["accuracy"] += float((logits.detach().argmax(-1) == actions).float().mean().item())
            seen += 1
            if config.log_every and seen % config.log_every == 0:
                log.info(
                    "epoch %d step %d/%d loss=%.4f action=%.4f acc=%.3f",
                    epoch + 1,
                    seen,
                    batches_per_epoch,
                    running["loss"] / seen,
                    running["action"] / seen,
                    running["accuracy"] / seen,
                )
        metrics = {k: v / max(1, seen) for k, v in running.items()}
        metrics["epoch"] = epoch + 1
        metrics["seconds"] = round(time.time() - epoch_start, 2)
        metrics["val"] = _validate(policy, dataset, val_episodes, config, device, rng)
        history.append(metrics)
        log.info(
            "epoch %d done: loss=%.4f acc=%.3f val=%s (%.1fs)",
            epoch + 1,
            metrics["loss"],
            metrics["accuracy"],
            metrics["val"],
            metrics["seconds"],
        )
        payload = {
            "config": policy_config.as_dict(),
            "model": policy.state_dict(),
            "action_masks": dataset.action_masks,
            "train_config": asdict(config),
            "metrics": metrics,
            "history": history,
        }
        val_loss = metrics["val"].get("loss")
        if val_loss is not None and val_loss < best_val:
            best_val = float(val_loss)
            best_path = out_dir / "best.ckpt"
        torch.save(payload, best_path)
        torch.save(payload, out_dir / "last.ckpt")

    manifest = {
        "data": config.data,
        "epochs": config.epochs,
        "history": history,
        "policy_config": policy_config.as_dict(),
        "action_masks": dataset.action_masks,
        "best": str(best_path),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return best_path


def _class_weights(
    dataset: PackedDataset,
    episodes: list[int],
    config: BcConfig,
    device: torch.device,
) -> torch.Tensor | None:
    """Inverse-frequency class weights for the imitation loss.

    Held-right dominates a platformer demonstration: roughly half of all frames
    are "run right", and an unweighted cross-entropy is happy to predict it
    forever - including at the boss, where the expert is attacking.  Weighting by
    the inverse square root of the frequency (the usual compromise between
    uniform and fully inverse) stops the majority class from erasing the rare but
    decisive ones.
    """
    if not config.class_weighting or not episodes:
        return None
    counts = np.zeros(dataset.num_actions, dtype=np.float64)
    for episode in episodes:
        start, end = dataset._episode_slices[episode]
        counts += np.bincount(
            np.asarray(dataset.actions[start:end], dtype=np.int64), minlength=dataset.num_actions
        ).astype(np.float64)
    if counts.sum() <= 0:
        return None
    frequency = counts / counts.sum()
    present = frequency > 0
    weights = np.ones(dataset.num_actions, dtype=np.float32)
    weights[present] = (1.0 / frequency[present]) ** float(config.class_weight_power)
    weights[present] /= weights[present].mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _validate(
    policy: VptPolicy,
    dataset: PackedDataset,
    val_episodes: list[int],
    config: BcConfig,
    device: torch.device,
    rng: np.random.Generator,
    batches: int = 8,
) -> dict[str, float]:
    """Held-out episodes only: a random split of frames would leak."""
    if not val_episodes:
        return {}
    policy.eval()
    total = 0.0
    correct = 0.0
    count = 0
    with torch.no_grad():
        for _ in range(batches):
            batch = dataset.sample_batch(
                config.batch_size, stack=config.stack, episodes=val_episodes, rng=rng
            )
            images = torch.from_numpy(batch["images"]).to(device)
            states = torch.from_numpy(batch["states"]).to(device)
            actions = torch.from_numpy(batch["actions"]).to(device)
            out = policy(images, states if dataset.state_dim else None)
            logits = out["action_logits"][:, -1]
            total += float(F.cross_entropy(logits, actions).item())
            correct += float((logits.argmax(-1) == actions).float().sum().item())
            count += int(actions.numel())
    policy.train()
    return {"loss": total / batches, "accuracy": correct / max(1, count)}
