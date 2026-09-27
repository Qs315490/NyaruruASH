"""Inference runner (paper Algorithm 2).

Each agent keeps:
- a short-term window of the last w_s (obs, action) pairs,
- a long-term memory bank M_n of key-moment observations (append-only; the
  policy reads the last w_l),
- a stuck timer C_n that resets on every new key moment.

The loop halts when any timer reaches delta; the caller then collects the
trajectories and memory banks for retrieval and bootstrapping.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ash.models.ash_policy import AshPolicy
from ash.memory.kdm import KeyMomentModel

log = logging.getLogger(__name__)


@dataclass
class AgentState:
    agent_id: int
    obs: np.ndarray | None = None          # last observation (H, W, C) uint8
    short_obs: list[np.ndarray] = field(default_factory=list)
    short_act: list[int] = field(default_factory=list)
    memory: list[np.ndarray] = field(default_factory=list)   # key-moment obs
    trajectory: list[np.ndarray] = field(default_factory=list)  # all obs
    stuck_timer: int = 0
    steps: int = 0
    #: cluster ids matched earlier in this trajectory; K must only fire once
    #: per cluster, otherwise one persistent visual event resets the timer
    #: forever and the agent never reports stuck.
    seen_clusters: set[int] = field(default_factory=set)


class InferenceRunner:
    def __init__(
        self,
        env_factory: Any,           # () -> env with step()/observation()
        policy: AshPolicy,
        kdm: KeyMomentModel,
        *,
        w_s: int = 32,
        w_l: int = 8,
        image_size: int = 128,
        device: str = "cpu",
        max_steps: int = 20_000,
        key_moment_cooldown: int = 30,
    ) -> None:
        self.env_factory = env_factory
        self.policy = policy.to(device).eval()
        self.kdm = kdm
        self.w_s = w_s
        self.w_l = w_l
        self.image_size = image_size
        self.device = torch.device(device)
        self.max_steps = max_steps
        # Debounce: the same visual event persists for many consecutive frames
        # (a cutscene, a menu, a room interior).  Without a cooldown every such
        # frame re-classifies as "new" and the stuck timer never advances.
        self.key_moment_cooldown = key_moment_cooldown

    # ------------------------------------------------------------------

    @staticmethod
    def _frame(result: Any) -> np.ndarray:
        """Accept Obs, StepResult, or a raw array; always return uint8 RGB."""
        obs = getattr(result, "obs", result)
        rgb = getattr(obs, "rgb", obs)
        return np.asarray(rgb, dtype=np.uint8)

    def _prep(self, obs: np.ndarray) -> np.ndarray:
        """(H, W, C) uint8 -> model input size, RGB, float [0,1]."""
        import cv2

        f = cv2.resize(obs, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if f.ndim == 2:
            f = np.stack([f] * 3, axis=-1)
        return f.astype(np.float32) / 255.0

    def _act(self, states: list[AgentState]) -> list[int]:
        """One batched policy forward across all live agents."""
        b = len(states)
        # short-term history: left-pad with the current frame repeated for the
        # first steps of an episode (no history yet) and the previous actions
        ws = self.w_s
        obs_hist = np.zeros((b, ws, self.image_size, self.image_size, 3), dtype=np.float32)
        act_hist = np.zeros((b, ws, self.policy.config.num_actions), dtype=np.float32)
        for i, s in enumerate(states):
            # history is [prev pairs..., (current obs, no-op slot)]: the policy
            # predicts the action FOR the current observation, so the current
            # frame must be the last observation token in the sequence.
            hist_obs = s.short_obs[-(ws - 1):] + [s.obs]
            hist_act = s.short_act[-(ws - 1):] + [0]
            pad = ws - len(hist_obs)
            for k, (o, a) in enumerate(zip(hist_obs, hist_act)):
                obs_hist[i, pad + k] = self._prep(o)
                act_hist[i, pad + k, a] = 1.0
            if 0 < pad < ws:
                obs_hist[i, :pad] = obs_hist[i, pad]  # repeat earliest frame
        mem = np.zeros((b, self.w_l, self.image_size, self._image_w(), 3), dtype=np.float32)
        for i, s in enumerate(states):
            m = s.memory[-self.w_l:]
            pad = self.w_l - len(m)
            for k, o in enumerate(m):
                mem[i, pad + k] = self._prep(o)
            if 0 < pad < self.w_l:
                mem[i, :pad] = mem[i, pad]
        ft = torch.from_numpy(obs_hist).to(self.device)
        at = torch.from_numpy(act_hist).to(self.device)
        mt = torch.from_numpy(mem).to(self.device)
        actions = self.policy.act(ft, at, mt)
        return actions.cpu().tolist()

    def _image_w(self) -> int:
        return self.image_size

    # ------------------------------------------------------------------

    def run(self, envs: list[Any], delta: int, timeout_s: float = 8 * 3600) -> dict:
        """Run all agents until one is stuck (timer >= delta) or max_steps."""
        states = [AgentState(agent_id=i) for i in range(len(envs))]
        # Prime every agent with its first observation.
        for s, env in zip(states, envs):
            s.obs = self._frame(env.reset())
            s.trajectory.append(s.obs)
        started = time.time()
        stuck_seen = False
        while not stuck_seen and time.time() - started < timeout_s:
            # Batched action selection for all agents.
            actions = self._act(states)
            for s, env, a in zip(states, envs, actions):
                s.obs = self._frame(env.step(a))
                s.trajectory.append(s.obs)
                s.short_obs.append(s.obs)
                s.short_act.append(a)
                if len(s.short_obs) > self.w_s:
                    s.short_obs.pop(0)
                    s.short_act.pop(0)
                s.steps += 1
                # Key-moment detection on the new observation.
                if s.steps % 4 == 0:  # embed every 4th frame to keep up with 20 fps
                    emb = self.kdm.embedder.embed(s.obs[None])[0]
                    if self.kdm.classify(emb, s.seen_clusters):
                        s.memory.append(s.obs)
                        s.stuck_timer = 0
                        s.seen_clusters.add(self.kdm.cluster_of(emb))
                    else:
                        s.stuck_timer += 4
                else:
                    s.stuck_timer += 1
                if s.stuck_timer >= delta:
                    stuck_seen = True
            if any(s.steps >= self.max_steps for s in states):
                break
        return {
            "trajectories": [np.asarray(s.trajectory) for s in states],
            "memories": [list(s.memory) for s in states],
            "steps": [s.steps for s in states],
            "stuck": stuck_seen,
        }

    def _seen_clusters(self, state: AgentState) -> set[int]:
        """Clusters already matched earlier in this trajectory (O(1) read).

        classify() adds to state.seen_clusters as a side effect; this helper
        exists only for external callers that want a snapshot.
        """
        return set(state.seen_clusters)
