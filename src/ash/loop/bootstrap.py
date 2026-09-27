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


def pseudo_label_stats(dataset: dict[str, np.ndarray] | WindowDataset, num_actions: int) -> dict:
    """How concentrated is the IDM's labelling of one corpus video?

    Returns counts and the share of the single most common class.  The first
    column of every window is the no-predecessor pad row (all-zero one-hot, i.e.
    noop), so it is dropped: counting it would flatter the distribution with
    padding rather than measure the labels.  `dataset` may be a plain mapping of
    windowed arrays or a WindowDataset, whose `actions` view is small enough to
    build (unlike its frames).
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


def logit_diagnosis(pair_logits: np.ndarray | None) -> dict:
    """Why did the IDM label every frame the same way?

    A collapsed labelling has two very different causes, and they need opposite
    responses, so `majority_share` alone cannot tell them apart:

    - the input genuinely looks the same (more data will not help), or
    - the argmax is a per-class bias that no input can outvote (more data will
      not help either, but the fix is to stop training pi, not to collect more).

    Splitting the logits into how much each class moves as the frame changes
    (temporal std) versus how far apart the classes sit on average (bias spread)
    separates them.  Measured on a real IDM: temporal std 0.008 against a bias
    spread of 0.13 - the answer was a fixed class regardless of the frame.  This
    used to be hand-written as a throwaway probe twice; it belongs in the report.
    """
    x = np.asarray(pair_logits, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        return {}
    temporal = float(x.std(axis=0).mean())
    bias = float(x.mean(axis=0).std())
    return {
        "logit_temporal_std": temporal,
        "logit_bias_spread": bias,
        # >= 1 means the class the IDM picks is decided by its bias rather than
        # by the frame; near 0 means the input is what drives the answer.
        "bias_over_temporal": (bias / temporal) if temporal > 0 else None,
    }


def _prep_uint8(frames: np.ndarray, image_size: int) -> np.ndarray:
    """(N, H, W, C) uint8 -> (N, image_size, image_size, 3) uint8."""
    import cv2

    out = np.empty((len(frames), image_size, image_size, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        g = cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_AREA)
        if g.ndim == 2:
            g = np.stack([g] * 3, axis=-1)
        out[i] = g
    return out


def _prep(frames: np.ndarray, image_size: int) -> np.ndarray:
    """(N, H, W, C) uint8 -> float32 [0,1] at model resolution."""
    return _prep_uint8(frames, image_size).astype(np.float32) / 255.0


class WindowDataset:
    """Sliding windows over one trajectory, materialized one batch at a time.

    The windows used to be assembled with np.stack, which yields
    (n_win, w_s, H, W, 3) float32: for one 9405-frame corpus video at 128x128
    with w_s=32 that is 59 GB, and the memory bank added another 15 GB.  On a
    16 GB machine the full-corpus bootstrap could never finish at all - the
    measured behaviour was 18 minutes of one core at 100% and not a single
    artifact - so the small `corpus-live` subset had been hiding it.  Windows
    share their frames, so only the *index* of a window is worth storing.
    """

    def __init__(
        self,
        frames: np.ndarray,      # (T, H, W, 3) uint8 at model resolution
        masks: np.ndarray,       # (T, num_actions) float32
        mem_idx: np.ndarray,     # (n_win, w_l) int64, -1 = padding
        *,
        w_s: int,
        w_l: int,
        image_size: int,
        logits: np.ndarray | None = None,
    ) -> None:
        self.frames = frames
        self.masks = masks
        self.mem_idx = mem_idx
        self.w_s = w_s
        self.w_l = w_l
        self.image_size = image_size
        #: Raw (T-1, num_actions) IDM logits, kept for logit_diagnosis().
        self.logits = logits

    def __len__(self) -> int:
        return max(0, len(self.frames) - self.w_s + 1)

    def window_index(self) -> np.ndarray:
        """(n_win, w_s) frame index of every position of every window."""
        return np.arange(len(self))[:, None] + np.arange(self.w_s)[None, :]

    def __getitem__(self, key: str) -> np.ndarray:
        """The small per-window array (`actions`); the big two are refused.

        Refusing loudly matters more than convenience here: an accidental
        `ds["frames"]` used to be a silent 59 GB allocation attempt.
        """
        if key == "actions":
            return self.masks[self.window_index()]
        raise KeyError(
            "%r is not materialized: windows share their frames, and building the "
            "whole array costs tens of GB on a real corpus (use batch(idx))" % (key,)
        )

    def batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Windows at `idx`, as (B, w_s, H, W, 3) / (B, w_s, A) / (B, w_l, H, W, 3)."""
        i = np.asarray(idx, dtype=np.int64)
        win = i[:, None] + np.arange(self.w_s)[None, :]
        return {
            "frames": self.frames[win].astype(np.float32) / 255.0,
            "actions": self.masks[win],
            "memories": self._memories(i),
        }

    def _memories(self, i: np.ndarray) -> np.ndarray:
        out = np.zeros(
            (len(i), self.w_l, self.image_size, self.image_size, 3), dtype=np.float32
        )
        if self.w_l == 0 or self.mem_idx.size == 0:
            return out
        take = self.mem_idx[i]              # (B, w_l)
        valid = take >= 0
        if valid.any():
            out[valid] = self.frames[take[valid]].astype(np.float32) / 255.0
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
        embeddings: np.ndarray | None = None,
    ) -> WindowDataset:
        """IDM pseudo-actions + K memories for one observation-only trajectory.

        Returns a WindowDataset: the windows are not materialized (see the class
        docstring for why that is not an option on a real corpus).

        `embeddings` are the DINOv2 vectors for `obs`, passed in when the caller
        already has them - the retrieval index holds them for every corpus video.
        Re-embedding a video to recompute vectors that are already in memory is a
        whole ViT pass per video per round, and it cannot change the answer: the
        embedder is frozen and deterministic.
        """
        cfg = self.config
        dev = next(idm.parameters()).device
        T = len(obs)
        frames = _prep_uint8(obs, cfg.image_size)
        # IDM over all consecutive pairs, batched, and moved to the GPU chunk by
        # chunk: putting the whole video on the card at once is 2.6 GB of VRAM
        # for a 13419-frame video, on a card the game is also using.
        num_actions = idm.config.num_actions
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, T - 1, 64):
                chunk = min(64, T - 1 - i)
                a = torch.from_numpy(
                    frames[i : i + chunk].astype(np.float32) / 255.0).to(dev)
                b = torch.from_numpy(
                    frames[i + 1 : i + chunk + 1].astype(np.float32) / 255.0).to(dev)
                # The raw logits are kept, not just the argmax: they are T-1 x
                # num_actions floats (3 MB for a 13419-frame video) and they are
                # the only way to tell *why* a labelling collapsed - see
                # logit_diagnosis.
                chunks.append(idm(a, b).float().cpu().numpy())
        pair_logits = (
            np.concatenate(chunks, axis=0) if chunks
            else np.zeros((0, num_actions), dtype=np.float32)
        )
        # (T-1,) class indices; frame 0 has no predecessor, so its row stays
        # all-zero and argmax reads it as noop, which is the correct label.
        idx = np.zeros(T, dtype=np.int64)
        if len(pair_logits):
            idx[1:] = pair_logits.argmax(axis=1)
        masks_full = np.zeros((T, num_actions), dtype=np.float32)
        masks_full[np.arange(T), idx] = 1.0
        # Key-moment flags per frame, in one batched K query rather than a
        # per-frame Python loop (3.4 ms/frame measured while a live run was
        # competing for the CPU; ~1.2 ms/frame idle).
        if embeddings is None:
            embeddings = kdm.embedder.embed(obs)
        key_flags = kdm.classify_sequence(embeddings)
        n_windows = T - cfg.w_s + 1
        if n_windows <= 0:
            raise ValueError(f"trajectory too short: T={T} < w_s={cfg.w_s}")
        # Memory slots are frame indices, not copies of frames.
        mem_idx = np.full((n_windows, cfg.w_l), -1, dtype=np.int64)
        key_positions = np.where(key_flags)[0]
        for i in range(n_windows):
            before = key_positions[key_positions < i]         # strictly before window start
            take = before[-cfg.w_l :] if cfg.w_l else before[:0]
            if len(take):
                mem_idx[i, cfg.w_l - len(take) :] = take
            # i == 0 has no memories at all; leave the -1 padding rather than
            # reference a nonexistent index (the pad slot does not exist when
            # the window starts before the first key moment).
        return WindowDataset(
            frames, masks_full, mem_idx,
            w_s=cfg.w_s, w_l=cfg.w_l, image_size=cfg.image_size,
            logits=pair_logits,
        )

    # ------------------------------------------------------------------

    def update_policy(self, policy: AshPolicy, dataset: WindowDataset) -> dict:
        cfg = self.config
        dev = resolve_device(cfg.device)
        policy = policy.to(dev).train()
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
        n = len(dataset)
        idx = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
        holdout = max(1, int(0.1 * n))
        train_idx, val_idx = idx[holdout:], idx[:holdout]

        def to_tensors(chunk):
            b = dataset.batch(chunk)
            return (torch.from_numpy(b["frames"]).to(dev),
                    torch.from_numpy(b["actions"]).to(dev),
                    torch.from_numpy(b["memories"]).to(dev))

        best_val = float("inf")
        for epoch in range(cfg.policy_epochs):
            rng.shuffle(train_idx)
            totals = []
            for i in range(0, len(train_idx), cfg.batch_size):
                f, a, m = to_tensors(train_idx[i : i + cfg.batch_size])
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
                    f, a, m = to_tensors(val_idx[i : i + cfg.batch_size])
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
        corpus_embeddings: dict[str, np.ndarray] | None = None,
    ) -> dict:
        """Full bootstrap: refit K on retrieved corpus, update IDM on agent
        trajectories, update pi on IDM-labelled corpus videos.

        `retrieved_ids` is D^R - the videos step 2 selected.  Every corpus read
        below is restricted to it: the paper refits K and pi on *D^R*, not on
        the whole internet corpus, and an earlier version accepted the
        retrieved list and then read every video anyway, which made retrieval a
        no-op that still reported a plausible ranking.

        `corpus_embeddings` maps a video id to its DINOv2 matrix when the caller
        already has it (the retrieval index does).  Both corpus passes below
        otherwise re-embed every video - twice per round - to recompute vectors
        that are already in memory.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        if retrieved_ids is not None and not retrieved_ids:
            log.warning("D^R is empty: retrieval selected no corpus video, so K and "
                        "pi keep their previous state this round")
        cached = corpus_embeddings or {}

        def embed_of(vid: str, obs: np.ndarray) -> np.ndarray:
            """The corpus embeddings, reusing the index where it has them."""
            if vid in cached:
                return cached[vid]
            return kdm.embedder.embed(obs)

        # 1. K on D^R.  The corpus is observation-only, so every video is its own
        # trajectory id; the distinct-trajectory filter is what keeps a cluster
        # that only ever appears in one video from becoming a "key moment".
        emb_chunks, traj_ids = [], []
        for vid, obs in corpus_loader(ids=retrieved_ids):
            if len(obs) == 0:
                continue
            embs = embed_of(vid, obs)
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
                ds = self.build_policy_dataset(obs, idm, kdm, embed_of(vid, obs))
            except ValueError as e:
                log.warning("skip corpus video %s: %s", vid, e)
                continue
            stats = pseudo_label_stats(ds, idm.config.num_actions)
            stats["video"] = vid
            stats.update(logit_diagnosis(ds.logits))
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
