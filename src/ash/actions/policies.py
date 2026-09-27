"""Heuristic and random action policies used to collect data before training.

These are intentionally simple.  They exist so that (a) the pipeline has data to
train on from day one, and (b) exploration during collection is not pure noise,
which matters when the objective is a milestone you must physically reach.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ash.actions import ActionSpace, mask_from_buttons
from ash.memory.state import StateFramer


class ActionPolicy(Protocol):
    def reset(self, *, seed: int | None = None) -> None: ...

    def act(self, obs: Any, *, step: int) -> int: ...


@dataclass
class RandomPolicy:
    """Uniform sampling over the action space, with sticky actions.

    Sticky actions (holding the previous choice with probability p) are what make
    random exploration produce usable trajectories in a platformer: white noise
    every frame is physically unplayable, while runs of a few frames of 'right'
    actually move the character.
    """

    action_space: ActionSpace
    stickiness: float = 0.7
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _last: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._last = self.action_space.noop

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        self._last = self.action_space.noop

    def act(self, obs: Any, *, step: int) -> int:
        if self._rng.random() < self.stickiness:
            return self._last
        self._last = self._rng.choice(self.action_space.to_list())
        return self._last


@dataclass
class RightwardPolicy:
    """Run right, jump when stuck, and attack once movement stops.

    This is a scripted expert whose job is to produce demonstrations worth
    imitating.  Its earlier version attacked on a timer, which is the wrong
    prior: a trained policy that copies it reaches the boss door and never
    learns to fight, because the demos themselves mostly did not fight.  The
    fix is to make the scripted expert actually solve each obstacle: it attacks
    while it is blocked, which is precisely when a real game demands a fight.
    """

    action_space: ActionSpace
    jump_every: int = 12
    attack_every: int = 6
    jump_probability: float = 0.25
    attack_probability: float = 0.5
    seed: int | None = None
    blocked_patience: int = 6
    blocked_attack_fraction: float = 0.6
    _rng: random.Random = field(init=False, repr=False)
    _last_x: float | None = None
    _stuck: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        self._last_x = None
        self._stuck = 0

    def act(self, obs: Any, *, step: int) -> int:
        state = getattr(obs, "state", {}) or {}
        player = state.get("player") or {}
        x = player.get("x")
        if isinstance(x, (int, float)):
            if self._last_x is not None and abs(float(x) - float(self._last_x)) < 1e-6:
                self._stuck += 1
            else:
                self._stuck = 0
            self._last_x = float(x)

        buttons = ["right"]
        blocked = self._stuck >= self.blocked_patience
        if blocked or (
            step % self.jump_every == 0 and self._rng.random() < self.jump_probability
        ):
            buttons.append("jump")
        # Attack when progress has stalled: that is what a door or a boss looks
        # like from the outside, and it is the behaviour the demo set was missing.
        if blocked and self._rng.random() < self.blocked_attack_fraction:
            buttons.append("attack")
        elif step % self.attack_every == 0 and self._rng.random() < self.attack_probability:
            buttons.append("attack")
        if self._rng.random() < 0.05:
            buttons.append("dash")

        mask = mask_from_buttons(buttons)
        if mask not in self.action_space:
            mask = self.action_space.noop
        return mask


@dataclass
class MixturePolicy:
    """Switch between two policies on a schedule.

    Used to collect data that contains both directed and exploratory behaviour:
    a dataset that is 100% one heuristic teaches the model only that heuristic.
    """

    primary: Any
    secondary: Any
    primary_fraction: float = 0.7
    segment_steps: int = 240
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _use_primary: bool = True

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        self._use_primary = self._rng.random() < self.primary_fraction
        for policy in (self.primary, self.secondary):
            policy.reset(seed=self._rng.randrange(1 << 30))

    def act(self, obs: Any, *, step: int) -> int:
        if self.segment_steps > 0 and step % self.segment_steps == 0:
            self._use_primary = self._rng.random() < self.primary_fraction
        policy = self.primary if self._use_primary else self.secondary
        return policy.act(obs, step=step)


@dataclass
class TorchPolicyActor:
    """Wrap a trained checkpoint as an ActionPolicy.

    The actor owns the frame stack, because the policy itself is stateless: it
    sees a fixed window of frames and predicts one action, which is what makes
    batched training simple and inference cheap.
    """

    policy: Any
    action_space: ActionSpace
    stack: int = 4
    device: str = "cpu"
    sample: bool = False
    temperature: float = 1.0
    image_size: int = 0
    _frames: list[np.ndarray] = field(default_factory=list, repr=False)
    _vectors: list[list[float]] = field(default_factory=list, repr=False)
    _framer: StateFramer | None = field(default=None, repr=False)
    _torch: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._frames = []
        self._vectors = []
        self._framer = StateFramer()
        if not self.image_size:
            self.image_size = int(getattr(self.policy.config, "image_size", 0) or 0)

    def _prepare(self, rgb: np.ndarray) -> np.ndarray:
        """Match the observation size the checkpoint was trained at.

        The game renders at its own resolution and the recorder may have downscaled
        it; a policy is only valid for the size it saw during training, so the
        conversion belongs here rather than in the environment.
        """
        if not self.image_size or rgb.shape[0] == self.image_size:
            return rgb
        import cv2

        return cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    def reset(self, *, seed: int | None = None) -> None:
        self._frames = []
        self._vectors = []
        if self._framer is not None:
            self._framer.reset()

    def _stack(self) -> tuple[Any, Any]:
        frames = list(self._frames)
        vectors = list(self._vectors)
        while len(frames) < self.stack:
            frames.insert(0, frames[0])
        while len(vectors) < self.stack:
            vectors.insert(0, vectors[0] if vectors else [0.0] * self._framer.dim)
        images = self._torch.as_tensor(np.stack(frames[-self.stack :]), dtype=self._torch.uint8)[None]
        state_tensor = None
        if getattr(self.policy.config, "state_dim", 0):
            rows = np.asarray(vectors[-self.stack :], dtype=np.float32)
            state_tensor = self._torch.as_tensor(rows)[None]
        return images, state_tensor

    def warmup(self, obs: Any, *, repeats: int | None = None) -> None:
        """Fill the frame stack from a real observation.

        An empty stack is padded by repeating the first frame, which is exactly
        the "no history" input the network never sees during training - that
        mismatch is what made an otherwise accurate policy stand still.  The
        single frame is repeated only to reach the required depth; the network
        reads the newest slice, and a fresh episode has no earlier frames.
        """
        if self._torch is None:
            import torch

            self._torch = torch
        if self._frames:
            return
        depth = int(repeats or self.stack)
        frame = self._prepare(np.asarray(obs.rgb))
        if self._framer is None:
            self._vectors = [[0.0]] * depth
        else:
            # Same call order as training: build the vector, then note the action.
            vector = self._framer.vector(dict(getattr(obs, "state", {}) or {}))
            self._vectors = [list(vector) for _ in range(depth)]
        self._frames = [frame.copy() for _ in range(depth)]

    def act(self, obs: Any, *, step: int) -> int:
        if self._torch is None:
            import torch

            self._torch = torch
        self.warmup(obs)
        self._frames.append(self._prepare(np.asarray(obs.rgb)))
        if self._framer is not None:
            self._vectors.append(self._framer.vector(dict(getattr(obs, "state", {}) or {})))
        if len(self._frames) > self.stack:
            self._frames = self._frames[-self.stack :]
            self._vectors = self._vectors[-self.stack :]
        images, state_tensor = self._stack()
        with self._torch.no_grad():
            out = self.policy.act(
                images.to(self.device),
                state_tensor.to(self.device) if state_tensor is not None else None,
                sample=self.sample,
                temperature=self.temperature,
            )
        index = int(out["action"].reshape(-1)[0].item())
        masks = self.action_space.to_list()
        if self._framer is not None:
            self._framer.note_action(int(masks[min(index, len(masks) - 1)]))
        return int(masks[min(index, len(masks) - 1)])


def action_space_from_masks(masks: list[int]) -> ActionSpace:
    """Rebuild the exact action space a checkpoint was trained with."""
    if not masks:
        return ActionSpace.minimal()
    return ActionSpace(masks=tuple(int(m) for m in masks), meta=dict.fromkeys(masks, "checkpoint"))
