"""Inference runner (paper Algorithm 2).

Each agent keeps:
- a short-term window of the last w_s (obs, action) pairs,
- a long-term memory bank M_n of key-moment observations (append-only; the
  policy reads the last w_l),
- a stuck timer C_n that resets on every new key moment.

The loop halts when any timer reaches delta; the caller then collects the
trajectories and memory banks for retrieval and bootstrapping.

Action plumbing (three distinct representations, do not conflate them):

    policy class index   what the head emits (0 .. len(action_space)-1)
    button mask          what env.step() consumes (a packed int)
    button vector        what the IDM predicts (multi-hot over BUTTONS)

`action_space.mask_at(index)` is the only sanctioned index -> mask conversion;
an earlier version passed the raw index to step(), which silently pressed an
unrelated button because the index is not the mask.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ash.actions.space import ActionSpace
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
    #: Button masks actually executed, parallel to trajectory[1:].  The IDM is
    #: retrained on the agents' own transitions with these as ground-truth
    #: labels (paper Algorithm 4 step 2), so they are recorded, not
    #: reconstructed from the policy indices afterwards.
    actions: list[int] = field(default_factory=list)
    stuck_timer: int = 0
    steps: int = 0
    #: cluster ids matched earlier in this trajectory; K must only fire once
    #: per cluster, otherwise one persistent visual event resets the timer
    #: forever and the agent never reports stuck.
    seen_clusters: set[int] = field(default_factory=set)
    #: K's verdicts for this round.  A round that reports no key moment has two
    #: very different explanations - every evaluation landed in noise (the live
    #: view is outside the corpus' feature space) or they all landed in one
    #: cluster already seen (the agent is looking at the same thing all round) -
    #: and the report has to say which one it was.
    #: Map id sampled during the round (every `map_sample_every` steps when the
    #: environment can report one).  Without it "did the agent get anywhere?"
    #: is unanswerable from the report: a round that never left the starting
    #: room and one that crossed three maps look identical.
    maps: list[int] = field(default_factory=list)
    k_evals: int = 0
    k_noise: int = 0
    k_in_key_cluster: int = 0
    k_fired: int = 0


def _is_menu_scene(env: Any) -> bool:
    """True when the backend says the current scene is a known menu template.

    The list lives in agent.js (V.MENU_SCENES); backends that do not implement
    the hook (the fake one) are never treated as menus."""
    try:
        info = env.safety() if hasattr(env, "safety") else {}
    except Exception:  # noqa: BLE001 - an unreadable scene must abort, not escape
        return False
    return bool(info.get("menu"))


class InferenceRunner:
    def __init__(
        self,
        env_factory: Any,           # () -> env with step()/observation()
        policy: AshPolicy,
        kdm: KeyMomentModel,
        *,
        action_space: ActionSpace,
        w_s: int = 32,
        w_l: int = 8,
        image_size: int = 128,
        device: str = "cpu",
        max_steps: int = 20_000,
        key_moment_cooldown: int = 30,
        #: How many times one round may back out of a menu with cancel, and how
        #: many cancel presses each attempt gets.  Both bounded: a scene that
        #: keeps reappearing is a loop, and a round must end rather than spin.
        max_menu_escapes: int = 8,
        max_menu_presses: int = 4,
        frame_skip: int = 1,
        max_confirm_presses: int = 3,
    ) -> None:
        #: Upper bound on ok presses spent clearing an input-gated scene.  Bounded
        #: because a screen that does not clear after a couple of oks is not a
        #: transition screen, and pressing on would be exactly the "press and
        #: hope" behaviour the gate exists to prevent.
        self.max_confirm_presses = max(0, int(max_confirm_presses))
        self.env_factory = env_factory
        self.policy = policy.to(device).eval()
        self.kdm = kdm
        self.action_space = action_space
        self.w_s = w_s
        self.w_l = w_l
        self.image_size = image_size
        self.device = torch.device(device)
        self.max_steps = max_steps
        #: Game frames one action advances.  This is the agent's timestep and it
        #: must equal the corpus sampling interval, because the IDM is trained
        #: on agent pairs and applied to corpus pairs.  It used to be 1 frame
        #: (1/60 s) against a 2 s corpus: a 120x mismatch that made the IDM
        #: label 99.8% of corpus frames with one constant class.
        self.frame_skip = max(1, int(frame_skip))
        # Debounce: the same visual event persists for many consecutive frames
        # (a cutscene, a menu, a room interior).  Without a cooldown every such
        # frame re-classifies as "new" and the stuck timer never advances.
        self.key_moment_cooldown = key_moment_cooldown
        self.max_menu_escapes = max(0, int(max_menu_escapes))
        self.max_menu_presses = max(1, int(max_menu_presses))

    # ------------------------------------------------------------------

    @staticmethod
    def _scene_gate(envs: list[Any]) -> tuple[str, str | None]:
        """Classify the current scene: "ok", "confirm", "menu" or "abort".

        Backends that do not implement the hook (the fake one) are always "ok".
        The CDP backend implements it and fails closed on an unreadable scene,
        because "I could not tell" must not mean "go ahead and press jump".

        "confirm" is a small allow-list of input-gated screens (the game's own
        transition scene) where one ok press is required and commits nothing.
        Refusing input there does not protect anything - it parked a live game
        in that scene permanently.  Everything else that is not Scene_Map
        aborts, because pressing keys there commits menu selections.
        """
        for env in envs:
            check = getattr(env, "unsafe_reason", None)
            if check is None:
                continue
            reason = check()
            if not reason:
                continue
            probe = getattr(env, "is_confirm_scene", None)
            if probe is not None and probe():
                return "confirm", reason
            return "menu" if _is_menu_scene(env) else "abort", reason
        return "ok", None

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
        """One batched policy forward; returns a button mask per agent."""
        b = len(states)
        # short-term history: left-pad with the current frame repeated for the
        # first steps of an episode (no history yet) and the previous actions
        ws = self.w_s
        n_act = self.policy.config.num_actions
        obs_hist = np.zeros((b, ws, self.image_size, self.image_size, 3), dtype=np.float32)
        act_hist = np.zeros((b, ws, n_act), dtype=np.float32)
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
        mem = np.zeros((b, self.w_l, self.image_size, self.image_size, 3), dtype=np.float32)
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
        indices = self.policy.act(ft, at, mt)
        # Policy index -> button mask: the environment has no notion of the
        # policy's class indices.
        return [self.action_space.mask_at(int(i)) for i in indices.cpu().tolist()]

    # ------------------------------------------------------------------

    def random_rollout(self, env: Any, steps: int, seed: int = 0) -> dict:
        """Uniformly random actions, run purely to widen the IDM's coverage.

        The paper updates the IDM on the agents' trajectories *supplemented with
        random-policy samples*.  Without them the IDM only ever sees the few,
        heavily biased transitions the current (bad) policy produces, and cannot
        label corpus frames whose dynamics it has never observed - which shows
        up as the IDM answering every corpus pair with one constant class.

        `seed` is the round's own: a fixed one made every round replay the SAME
        action sequence, so the "supplement" re-explored the same hundred steps
        forever instead of widening coverage.  Within a round it stays
        deterministic, which is what a replay needs.

        The safety gate applies here exactly as it does to the policy round:
        random keystrokes on a menu screen are still menu selections.
        """
        rng = np.random.default_rng(int(seed))
        obs = self._frame(env.reset())
        frames = [obs]
        acts: list[int] = []
        confirms = 0
        menu_escapes = 0
        for _ in range(max(0, int(steps))):
            kind, reason = self._scene_gate([env])
            if kind == "menu":
                if self._leave_menu([env], menu_escapes):
                    menu_escapes += 1
                    continue
                log.error("aborting random-policy rollout: menu did not clear: %s", reason)
                break
            if kind == "confirm":
                confirms += 1
                if confirms > self.max_confirm_presses:
                    log.error("aborting random-policy rollout: %s", reason)
                    break
                press = getattr(env, "press_ok", None)
                if press is not None:
                    press()
                continue
            if kind == "abort":
                log.error("aborting random-policy rollout: %s", reason)
                break
            confirms = 0
            idx = int(rng.integers(len(self.action_space)))
            obs = self._frame(env.step(self.action_space.mask_at(idx), frames=self.frame_skip))
            frames.append(obs)
            acts.append(idx)
        return {"obs": np.asarray(frames), "act": np.asarray(acts, dtype=np.int64)}

    def run(
        self,
        envs: list[Any],
        delta: int,
        timeout_s: float = 8 * 3600,
        random_steps: int = 0,
        random_seed: int = 0,
    ) -> dict:
        """Run all agents until one is stuck (timer >= delta) or max_steps."""
        states = [AgentState(agent_id=i) for i in range(len(envs))]
        # Prime every agent with its first observation.
        for s, env in zip(states, envs):
            s.obs = self._frame(env.reset())
            s.trajectory.append(s.obs)
        started = time.time()
        stuck_seen = False
        abort_reason: str | None = None
        confirms = 0
        menu_escapes = 0
        while not stuck_seen and time.time() - started < timeout_s:
            # Safety gate, before a single key is dispatched.  In RPG Maker the
            # ok/cancel keys ARE the policy's jump/attack keys, so acting on a
            # title or menu screen commits menu selections: an unattended run
            # pressed Z on the title screen and loaded the player's save.
            kind, reason = self._scene_gate(envs)
            if kind == "menu":
                if self._leave_menu(envs, menu_escapes):
                    menu_escapes += 1
                    continue
                # Falling through here would dispatch gameplay keys inside the
                # menu - the exact thing the gate exists to prevent.
                abort_reason = "menu scene did not clear with cancel: %s" % reason
                log.error("aborting inference round: %s", abort_reason)
                break
            if kind == "confirm":
                confirms += 1
                if confirms > self.max_confirm_presses:
                    abort_reason = (
                        "confirm scene %s did not clear after %d ok presses"
                        % (reason, self.max_confirm_presses)
                    )
                    log.error("aborting inference round: %s", abort_reason)
                    break
                log.info("pressing ok once to clear a confirm scene (%d/%d): %s",
                         confirms, self.max_confirm_presses, reason)
                for env in envs:
                    press = getattr(env, "press_ok", None)
                    if press is not None:
                        press()
                continue
            if kind == "abort":
                abort_reason = reason
                log.error("aborting inference round: %s", abort_reason)
                break
            confirms = 0
            # Batched action selection for all agents.
            masks = self._act(states)
            for s, env, mask in zip(states, envs, masks):
                # frame_skip game frames per action: the agent's timestep must
                # match the corpus sampling interval, or the IDM is trained on
                # one time scale and applied at another.
                s.obs = self._frame(env.step(mask, frames=self.frame_skip))
                s.trajectory.append(s.obs)
                # Record the class index, not the mask: the IDM is trained on
                # these as labels and its head is a classifier over the action
                # space.  The mask is what the env consumes, the index is what
                # the models speak; mixing them up silently mislabels every
                # training pair (mask 3 is not class 3).
                index = self.action_space.index_of(mask)
                s.actions.append(index)
                s.short_obs.append(s.obs)
                s.short_act.append(index)
                if len(s.short_obs) > self.w_s:
                    s.short_obs.pop(0)
                    s.short_act.pop(0)
                s.steps += 1
                if s.steps % 10 == 0:
                    self._sample_map(env, s)
                # Key-moment detection on every observation.  A step is now
                # control_interval_s of game time, so the stuck timer counts
                # *steps without a new key moment* - which is what the paper's
                # Delta means - instead of the old mix of frames and steps.
                emb = self.kdm.embedder.embed(s.obs[None])[0]
                fired, label, is_key_cluster = self.kdm.observe(emb, s.seen_clusters)
                s.k_evals += 1
                if label < 0:
                    s.k_noise += 1
                elif is_key_cluster:
                    s.k_in_key_cluster += 1
                if fired:
                    s.memory.append(s.obs)
                    s.stuck_timer = 0
                    s.seen_clusters.add(label)
                    s.k_fired += 1
                else:
                    s.stuck_timer += 1
                if s.stuck_timer >= delta:
                    stuck_seen = True
            if any(s.steps >= self.max_steps for s in states):
                break
        # The paper supplements the IDM's agent transitions with random-policy
        # samples; without them the IDM only sees what the current policy
        # happened to do and cannot label unseen dynamics.
        random_trajectories: list[dict] = []
        if random_steps > 0 and not abort_reason:
            for env in envs:
                traj = self.random_rollout(env, random_steps, random_seed)
                if len(traj["act"]):
                    random_trajectories.append(traj)
            log.info(
                "random-policy rollout: %d transitions",
                sum(len(t["act"]) for t in random_trajectories),
            )
        # Each trajectory is a dict, not a bare array: bootstrap needs the
        # executed masks alongside the frames to retrain the IDM, and the masks
        # cannot be recovered from the frames afterwards.
        return {
            "trajectories": [
                {
                    "obs": np.asarray(s.trajectory),
                    "act": np.asarray(s.actions, dtype=np.int64),
                }
                for s in states
            ],
            "random_trajectories": random_trajectories,
            "maps": [list(s.maps) for s in states],
            "memories": [list(s.memory) for s in states],
            "key_stats": [
                {"evals": s.k_evals, "noise": s.k_noise,
                 "in_key_cluster": s.k_in_key_cluster, "fired": s.k_fired}
                for s in states
            ],
            "steps": [s.steps for s in states],
            "stuck": stuck_seen,
            #: Non-None when the round stopped for safety instead of getting
            #: stuck; the orchestrator records it so a truncated round is
            #: never mistaken for a converged one.
            "aborted": abort_reason,
        }

    def _sample_map(self, env: Any, state: AgentState) -> None:
        """Record the current map id, when the backend can report one.

        Best-effort on purpose: the loop must not depend on a diagnostic, and a
        backend without state() simply reports nothing.
        """
        read = getattr(env, "state", None)
        if read is None:
            return
        try:
            player = (read() or {}).get("player") or {}
        except Exception:  # noqa: BLE001 - a diagnostic must never fail a round
            return
        map_id = player.get("mapId")
        if map_id is not None and (not state.maps or state.maps[-1] != map_id):
            state.maps.append(int(map_id))

    def _leave_menu(self, envs: list[Any], escapes: int) -> bool:
        """Try the cancel-only menu escape.  True when gameplay is reached.

        The escape is bounded per round: a scene that keeps coming back is a
        loop, not an accident, and the round must end rather than spin.
        """
        if escapes >= self.max_menu_escapes:
            return False
        for env in envs:
            escape = getattr(env, "escape_menu", None)
            if escape is None:
                return False
            try:
                result = escape(max_presses=self.max_menu_presses)
            except Exception as exc:  # noqa: BLE001 - a failed escape aborts the round
                log.error("menu escape failed: %s", str(exc)[:160])
                return False
            log.warning("escaped a menu with %d cancel press(es): %s",
                        result.get("presses"), result.get("scene"))
            return bool(result.get("escaped"))
        return False

    def _seen_clusters(self, state: AgentState) -> set[int]:
        """Clusters already matched earlier in this trajectory (O(1) read).

        classify() adds to state.seen_clusters as a side effect; this helper
        exists only for external callers that want a snapshot.
        """
        return set(state.seen_clusters)
