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
from ash.utils.device import resolve_device
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
    #: None/"auto" picks cuda when available; see ash.utils.device.
    device: str | None = None
    #: VRAM guard: the game shares the 12 GB card; keep batches modest.
    grad_clip: float = 1.0
    #: Refuse to train pi when one pseudo-action class covers more than this
    #: share of the corpus windows.
    #:
    #: A bias-dominated IDM answers every pair with the same class - measured,
    #: not assumed: its per-frame logit variance was 0.008 against a 0.13 class
    #: prior, so its argmax never moved with the input.  Training pi on that
    #: teaches it to emit one constant action, and the resulting loss
    #: (policy_val ~1e-6) reads exactly like convergence.  Skipping the update
    #: and saying so is strictly better than destroying the policy quietly.
    max_pseudo_majority: float = 0.9


def pseudo_label_stats(dataset: dict[str, np.ndarray], num_actions: int) -> dict:
    """How concentrated is the IDM's labelling of one corpus video?

    Returns counts and the share of the single most common class.  The first
    column of every window is the no-predecessor pad row (all-zero one-hot, i.e.
    noop), so it is dropped: counting it would flatter the distribution with
    padding rather than measure the labels.
    """
    actions = np.asarray(dataset["actions"])
    if actions.ndim != 3:
        raise ValueError("actions must be (windows, w_s, num_actions)")
    if actions.shape[1] > 1:
        actions = actions[:, 1:, :]
    labels = actions.argmax(axis=-1).ravel()
    if labels.size == 0:
        return {"labels": 0, "classes_used": 0, "majority_share": 1.0,
                "entropy": 0.0, "uniform_entropy": float(np.log(num_actions))}
    counts = np.bincount(labels, minlength=num_actions).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return {
        "labels": int(labels.size),
        "classes_used": int((counts > 0).sum()),
        "majority_share": float(counts.max() / counts.sum()),
        "entropy": float(-(p * np.log(p)).sum()),
        "uniform_entropy": float(np.log(num_actions)),
    }


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

    def update_idm(self, idm: IdmModel, obs: np.ndarray, action_index: np.ndarray) -> dict:
        """Train the IDM on explicit (before, after, action) transitions.

        The IDM is a classifier over the action space, not a multi-label key
        predictor: the ASH paper's pseudo-actions are one-hot vectors used
        directly as policy labels, so the loss is cross-entropy on a single
        class.

        The transition list is built first and indexed by *pair* afterwards, so
        the label of a pair can never drift onto its neighbour.  An earlier
        version indexed `action_index[chunk + 1]` on a frame-indexed array: it
        labelled every pair with the *following* action and ran off the end of
        the array, which is the shape of bug that trains happily and simply
        learns the wrong thing.
        """
        obs = np.asarray(obs)
        action_index = np.asarray(action_index, dtype=np.int64)
        before, after = obs[:-1], obs[1:]
        if len(action_index) != len(before):
            raise ValueError(
                "action_index must hold one label per transition: expected %d "
                "for %d frames, got %d" % (len(before), len(obs), len(action_index))
            )
        return self._fit_idm(idm, before, after, action_index)

    def _fit_idm(
        self,
        idm: IdmModel,
        before: np.ndarray,
        after: np.ndarray,
        action_index: np.ndarray,
    ) -> dict:
        """Shared trainer: aligned (before, after) frames and their labels."""
        cfg = self.config
        dev = resolve_device(cfg.device)
        n = len(before)
        if n < 2:
            return {"idm_skipped": True}
        idm = idm.to(dev).train()
        opt = torch.optim.AdamW(idm.parameters(), lr=cfg.lr)
        idx = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
        holdout = max(1, int(0.1 * n))
        train_idx, val_idx = idx[holdout:], idx[:holdout]
        ce = nn.CrossEntropyLoss()
        best_val = float("inf")
        for epoch in range(cfg.idm_epochs):
            rng.shuffle(train_idx)
            totals = []
            for i in range(0, len(train_idx), cfg.batch_size):
                chunk = train_idx[i : i + cfg.batch_size]
                a = torch.from_numpy(_prep(before[chunk], cfg.image_size)).to(dev)
                b = torch.from_numpy(_prep(after[chunk], cfg.image_size)).to(dev)
                y = torch.from_numpy(action_index[chunk]).to(dev)
                loss = ce(idm(a, b), y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(idm.parameters(), cfg.grad_clip)
                opt.step()
                totals.append(loss.item())
            with torch.inference_mode():
                v = []
                for i in range(0, len(val_idx), cfg.batch_size):
                    chunk = val_idx[i : i + cfg.batch_size]
                    a = torch.from_numpy(_prep(before[chunk], cfg.image_size)).to(dev)
                    b = torch.from_numpy(_prep(after[chunk], cfg.image_size)).to(dev)
                    y = torch.from_numpy(action_index[chunk]).to(dev)
                    v.append(ce(idm(a, b), y).item())
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
        # IDM over all consecutive pairs, batched.  The output is a class index
        # per frame; the policy's action token is the one-hot of that index
        # (the paper's "pseudo-actions, encoded as one-hot vectors").
        num_actions = idm.config.num_actions
        pseudo = []
        with torch.inference_mode():
            for i in range(0, T - 1, 64):
                chunk = min(64, T - 1 - i)
                logits = idm(pt[i : i + chunk], pt[i + 1 : i + chunk + 1])
                pseudo.append(logits.argmax(dim=1).cpu().numpy())
        # (T-1,) class indices; frame 0 has no predecessor, so its row stays
        # all-zero and argmax reads it as noop, which is the correct label.
        idx = np.zeros(T, dtype=np.int64)
        if pseudo:
            idx[1:] = np.concatenate(pseudo, axis=0)
        masks_full = np.zeros((T, num_actions), dtype=np.float32)
        masks_full[np.arange(T), idx] = 1.0
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
        dev = resolve_device(cfg.device)
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
        retrieved_ids: list[str] | None = None,
        random_trajectories: list[np.ndarray] | None = None,
    ) -> dict:
        """Full bootstrap: refit K on retrieved corpus, update IDM on agent
        trajectories, update pi on IDM-labelled corpus videos.

        `retrieved_ids` is D^R - the videos step 2 selected.  Every corpus read
        below is restricted to it: the paper refits K and pi on *D^R*, not on
        the whole internet corpus, and an earlier version accepted the
        retrieved list and then read every video anyway, which made retrieval a
        no-op that still reported a plausible ranking.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        if retrieved_ids is not None and not retrieved_ids:
            log.warning("D^R is empty: retrieval selected no corpus video, so K and "
                        "pi keep their previous state this round")
        # 1. K on D^R.  The corpus is observation-only, so every video is its own
        # trajectory id; the distinct-trajectory filter is what keeps a cluster
        # that only ever appears in one video from becoming a "key moment".
        emb_chunks, traj_ids = [], []
        for vid, obs in corpus_loader(ids=retrieved_ids):
            if len(obs) == 0:
                continue
            embs = kdm.embedder.embed(obs)
            emb_chunks.append(embs)
            traj_ids.extend([vid] * len(embs))
        kdm_report = {"kdm_skipped": True}
        if emb_chunks:
            kdm_report = self.update_kdm(kdm, np.concatenate(emb_chunks), traj_ids)
        # 2. IDM on agent trajectories (real action labels from the runner),
        # supplemented with the random-policy samples the runner collected.
        # Without the supplement the IDM only ever sees the narrow, biased slice
        # of dynamics the current policy produces, and answers corpus frames
        # with one constant class instead of a distribution.
        idm_data = list(trajectories) + list(random_trajectories or [])
        n_frames = sum(len(t["obs"]) for t in idm_data)
        idm_report = {"idm_skipped": True}
        if n_frames > 2:
            idm_report = self.update_idm_from_trajectories(idm, idm_data)
            idm_report["agent_transitions"] = sum(
                max(0, len(t["obs"]) - 1) for t in trajectories
            )
            idm_report["random_transitions"] = sum(
                max(0, len(t["obs"]) - 1) for t in (random_trajectories or [])
            )
        # 3. pi on D^R with pseudo-actions, but only where those labels carry
        # signal.  A constant target trains a constant policy and reports a
        # near-zero loss, so the degeneracy is checked before the update rather
        # than discovered later as "the agent does nothing".
        policy_reports = []
        label_stats: list[dict] = []
        for vid, obs in corpus_loader(ids=retrieved_ids):
            try:
                ds = self.build_policy_dataset(obs, idm, kdm)
            except ValueError as e:
                log.warning("skip corpus video %s: %s", vid, e)
                continue
            stats = pseudo_label_stats(ds, idm.config.num_actions)
            stats["video"] = vid
            label_stats.append(stats)
            if stats["majority_share"] > self.config.max_pseudo_majority:
                log.error(
                    "video %s: IDM pseudo-labels are %.1f%% one class (%d/%d classes "
                    "used, entropy %.3f of %.3f) - refusing to train pi on a constant "
                    "target; the IDM has not learned input-dependent dynamics yet",
                    vid, 100.0 * stats["majority_share"], stats["classes_used"],
                    idm.config.num_actions, stats["entropy"], stats["uniform_entropy"],
                )
                continue
            policy_reports.append(self.update_policy(policy, ds))
        save_policy(policy, out_dir / "policy.pt")
        save_idm(idm, out_dir / "idm.pt")
        return {
            "kdm": kdm_report,
            "idm": idm_report,
            "policy": policy_reports,
            "pseudo_labels": label_stats,
        }

    def update_idm_from_trajectories(self, idm: IdmModel, trajectories: list) -> dict:
        """Train the IDM on the agents' own transitions.

        Each trajectory is {"obs": (T,H,W,C), "act": (T-1,)} - one class index
        per executed step, so act[t] is the action that turned obs[t] into
        obs[t+1].

        Pairs are built *inside* each trajectory and only then concatenated.
        Concatenating the frames and the labels first would splice the last
        frame of one trajectory onto the first frame of the next and label that
        phantom transition with whatever action happened to sit at the seam.
        """
        before_parts, after_parts, act_parts = [], [], []
        for t in trajectories:
            obs = np.asarray(t["obs"])
            act = np.asarray(t["act"], dtype=np.int64)
            if len(obs) < 2:
                continue
            before_parts.append(obs[:-1])
            after_parts.append(obs[1:])
            act_parts.append(act)
        if not before_parts:
            return {"idm_skipped": True}
        before = np.concatenate(before_parts)
        after = np.concatenate(after_parts)
        act = np.concatenate(act_parts)
        if len(act) != len(before):
            raise ValueError(
                "each trajectory must carry one action per transition: "
                "%d transitions but %d actions" % (len(before), len(act))
            )
        # The IDM is a classifier over the policy's action space, so the label
        # is the class index itself - no one-hot expansion, no key unpacking.
        return self._fit_idm(idm, before, after, act)
