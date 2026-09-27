"""Bootstrapping (paper Algorithm 4): update K, IDM, and pi.

Order matters and follows the paper exactly:

1. K is refit on the retrieved corpus D^R (embeddings + all previously
   clustered embeddings), because the agent just encountered visuals the old
   clusters did not cover.
2. The IDM is updated on the agents' own trajectories (real action labels),
   supplemented with random-policy samples for coverage of new dynamics.
3. pi is updated on D^R with IDM pseudo-actions and K-constructed memories:
   for each window of w_s observations, the memory prefix is the w_l most
   recent key-moment frames *before* the window starts.

10% of samples are held out for both IDM and pi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ash.memory.kdm import KeyMomentModel
from ash.models.ash_policy import AshPolicy, save_policy
from ash.models.idm import IdmModel, save_idm

log = logging.getLogger(__name__)


@dataclass
class BootstrapConfig:
    w_s: int = 32
    w_l: int = 8
    image_size: int = 128
    idm_epochs: int = 3
    policy_epochs: int = 3
    batch_size: int = 8
    lr: float = 3e-5
    device: str = "cpu"
    #: VRAM guard: the game shares the 12 GB card; keep batches modest.
    grad_clip: float = 1.0


def _prep(frames: np.ndarray, image_size: int) -> np.ndarray:
    """(N, H, W, C) uint8 -> float32 [0,1] at model resolution."""
    import cv2

    out = np.empty((len(frames), image_size, image_size, 3), dtype=np.float32)
    for i, f in enumerate(frames):
        g = cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_AREA)
        if g.ndim == 2:
            g = np.stack([g] * 3, axis=-1)
        out[i] = g.astype(np.float32) / 255.0
    return out


class Bootstrapper:
    def __init__(self, config: BootstrapConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------

    def update_kdm(self, kdm: KeyMomentModel, corpus_embeddings: np.ndarray,
                   trajectory_ids: np.ndarray) -> dict:
        return kdm.fit(corpus_embeddings, trajectory_ids)

    def update_idm(self, idm: IdmModel, obs: np.ndarray, actions: np.ndarray) -> dict:
        """Pairs (obs[t], obs[t+1]) -> mask[t+1]; see ash.train.idm for details."""
        cfg = self.config
        dev = torch.device(cfg.device)
        idm = idm.to(dev).train()
        opt = torch.optim.AdamW(idm.parameters(), lr=cfg.lr)
        n = len(obs) - 1
        if n < 2:
            return {"idm_skipped": True}
        idx = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
        holdout = max(1, int(0.1 * n))
        train_idx, val_idx = idx[holdout:], idx[:holdout]
        bce = nn.BCEWithLogitsLoss()
        best_val = float("inf")
        for epoch in range(cfg.idm_epochs):
            rng.shuffle(train_idx)
            totals = []
            for i in range(0, len(train_idx), cfg.batch_size):
                chunk = train_idx[i : i + cfg.batch_size]
                a = torch.from_numpy(_prep(obs[chunk], cfg.image_size)).to(dev)
                b = torch.from_numpy(_prep(obs[chunk + 1], cfg.image_size)).to(dev)
                y = torch.from_numpy(actions[chunk + 1].astype(np.float32)).to(dev)
                loss = bce(idm(a, b), y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(idm.parameters(), cfg.grad_clip)
                opt.step()
                totals.append(loss.item())
            with torch.inference_mode():
                v = []
                for i in range(0, len(val_idx), cfg.batch_size):
                    chunk = val_idx[i : i + cfg.batch_size]
                    a = torch.from_numpy(_prep(obs[chunk], cfg.image_size)).to(dev)
                    b = torch.from_numpy(_prep(obs[chunk + 1], cfg.image_size)).to(dev)
                    y = torch.from_numpy(actions[chunk + 1].astype(np.float32)).to(dev)
                    v.append(bce(idm(a, b), y).item())
                val = float(np.mean(v)) if v else float("inf")
            log.info("idm epoch %d: train %.4f val %.4f", epoch, np.mean(totals), val)
            best_val = min(best_val, val)
        idm.eval()
        return {"idm_val": best_val}

    # ------------------------------------------------------------------

    def build_policy_dataset(
        self,
        obs: np.ndarray,               # (T, H, W, C) uint8, one trajectory
        idm: IdmModel,
        kdm: KeyMomentModel,
    ) -> dict[str, np.ndarray]:
        """IDM pseudo-actions + K memories for one observation-only trajectory.

        Returns dict with frames (T-ws+1, ws, H, W, C), actions, memories
        (T-ws+1, wl, H, W, C).
        """
        cfg = self.config
        dev = next(idm.parameters()).device
        T = len(obs)
        frames = _prep(obs, cfg.image_size)
        pt = torch.from_numpy(frames).to(dev)
        # IDM over all consecutive pairs, batched.
        masks = []
        with torch.inference_mode():
            for i in range(0, T - 1, 64):
                chunk = min(64, T - 1 - i)
                a = pt[i : i + chunk]
                b = pt[i + 1 : i + chunk + 1]
                logits = idm(a, b)
                masks.append((torch.sigmoid(logits) > 0.5).float().cpu().numpy())
        am = np.concatenate(masks, axis=0)                    # (T-1, num_actions)
        masks_full = np.zeros((T, am.shape[1]), dtype=np.float32)
        masks_full[1:] = am
        # Key-moment flags per frame.
        key_flags = np.zeros(T, dtype=bool)
        seen: set[int] = set()
        embedder = kdm.embedder
        embs = embedder.embed(obs)
        for t in range(T):
            if kdm.classify(embs[t], seen):
                key_flags[t] = True
                seen.add(self._cluster_of(kdm, embs[t]))
        # Assemble training windows.
        n_windows = T - cfg.w_s + 1
        if n_windows <= 0:
            raise ValueError(f"trajectory too short: T={T} < w_s={cfg.w_s}")
        win_frames = np.stack([frames[i : i + cfg.w_s] for i in range(n_windows)])
        win_masks = np.stack([masks_full[i : i + cfg.w_s] for i in range(n_windows)])
        win_mem = np.zeros((n_windows, cfg.w_l, cfg.image_size, cfg.image_size, 3), dtype=np.float32)
        key_positions = np.where(key_flags)[0]
        for i in range(n_windows):
            before = key_positions[key_positions < i]         # strictly before window start
            take = before[-cfg.w_l :]
            pad = cfg.w_l - len(take)
            for k, p in enumerate(take):
                win_mem[i, pad + k] = frames[p]
            # i == 0 has no memories at all; leave zeros rather than copying
            # a nonexistent index (the pad slot does not exist when pad == w_l).
        return {"frames": win_frames, "actions": win_masks, "memories": win_mem}

    def _cluster_of(self, kdm: KeyMomentModel, emb: np.ndarray) -> int:
        import hdbscan

        label, _ = hdbscan.approximate_predict(kdm._clusterer, emb.reshape(1, -1))
        return int(label[0])

    # ------------------------------------------------------------------

    def update_policy(self, policy: AshPolicy, dataset: dict) -> dict:
        cfg = self.config
        dev = torch.device(cfg.device)
        policy = policy.to(dev).train()
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
        frames = dataset["frames"]
        actions = dataset["actions"]
        memories = dataset["memories"]
        n = len(frames)
        idx = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
        holdout = max(1, int(0.1 * n))
        train_idx, val_idx = idx[holdout:], idx[:holdout]
        best_val = float("inf")
        for epoch in range(cfg.policy_epochs):
            rng.shuffle(train_idx)
            totals = []
            for i in range(0, len(train_idx), cfg.batch_size):
                chunk = train_idx[i : i + cfg.batch_size]
                f = torch.from_numpy(frames[chunk]).to(dev)
                a = torch.from_numpy(actions[chunk]).to(dev)
                m = torch.from_numpy(memories[chunk]).to(dev)
                logits = policy(f, a, m)
                # logits are (B, ws, num_actions); the target for every window
                # position is that position's IDM pseudo-action, not just the
                # last step - the paper sums the CE over all w_s positions.
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    a.reshape(-1, a.shape[-1]).argmax(dim=1),
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                opt.step()
                totals.append(loss.item())
            with torch.inference_mode():
                v = []
                for i in range(0, len(val_idx), cfg.batch_size):
                    chunk = val_idx[i : i + cfg.batch_size]
                    f = torch.from_numpy(frames[chunk]).to(dev)
                    a = torch.from_numpy(actions[chunk]).to(dev)
                    m = torch.from_numpy(memories[chunk]).to(dev)
                    logits = policy(f, a, m)
                    v.append(
                        nn.functional.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            a.reshape(-1, a.shape[-1]).argmax(dim=1),
                        ).item()
                    )
                val = float(np.mean(v)) if v else float("inf")
            log.info("policy epoch %d: train %.4f val %.4f", epoch, np.mean(totals), val)
            best_val = min(best_val, val)
        policy.eval()
        return {"policy_val": best_val}

    # ------------------------------------------------------------------

    def run(
        self,
        policy: AshPolicy,
        idm: IdmModel,
        kdm: KeyMomentModel,
        trajectories: list[np.ndarray],
        corpus_loader: Any,
        out_dir: Path,
    ) -> dict:
        """Full bootstrap: refit K on retrieved corpus, update IDM on agent
        trajectories, update pi on IDM-labelled corpus videos."""
        out_dir.mkdir(parents=True, exist_ok=True)
        # 1. K on D^R
        emb_chunks, traj_ids = [], []
        for vid, obs in corpus_loader(retrieved_only=True):
            embs = kdm.embedder.embed(obs)
            emb_chunks.append(embs)
            traj_ids.extend([vid] * len(embs))
        kdm_report = self.update_kdm(kdm, np.concatenate(emb_chunks), traj_ids)
        # 2. IDM on agent trajectories (real labels)
        all_obs = np.concatenate([t for t in trajectories if len(t) > 1])
        all_act = np.zeros((len(all_obs),), dtype=np.int64)  # placeholder; runner supplies real actions
        idm_report = {"idm_skipped": True}
        if len(all_obs) > 2:
            idm_report = self.update_idm_from_trajectories(idm, trajectories)
        # 3. pi on D^R with pseudo-actions
        policy_reports = []
        for vid, obs in corpus_loader(retrieved_only=True):
            try:
                ds = self.build_policy_dataset(obs, idm, kdm)
            except ValueError as e:
                log.warning("skip corpus video %s: %s", vid, e)
                continue
            policy_reports.append(self.update_policy(policy, ds))
        save_policy(policy, out_dir / "policy.pt")
        save_idm(idm, out_dir / "idm.pt")
        return {"kdm": kdm_report, "idm": idm_report, "policy": policy_reports}

    def update_idm_from_trajectories(self, idm: IdmModel, trajectories: list) -> dict:
        obs = np.concatenate([t["obs"] for t in trajectories if len(t["obs"]) > 1])
        act = np.concatenate([t["act"] for t in trajectories if len(t["obs"]) > 1])
        actions = np.zeros((len(obs), idm.config.num_keys), dtype=np.float32)
        for i, a in enumerate(act):
            if 0 <= a < idm.config.num_keys:
                actions[i, a] = 1.0
        return self.update_idm(idm, obs, actions)
