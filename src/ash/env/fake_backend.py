"""A deterministic toy platformer used for tests, CI and dry runs.

It is *not* the game.  It is a 2-D kinematic toy with a floor, a gap to jump, two
ledges, a boss and a goal seal.  Its value is that it implements the same
contract as the real backends - including bit-exact snapshot/restore - so every
other part of the pipeline (recorder, dataset, model, search, evaluator) can be
developed and regression-tested before the real game is installed.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ash.actions import ActionSpace, buttons_from_mask
from ash.env.base import Obs, StepResult

# World layout: (x_start, x_end, top_y)
_PLATFORMS: tuple[tuple[float, float, float], ...] = (
    (0.0, 40.0, 0.0),
    (44.0, 70.0, 8.0),
    (74.0, 118.0, 15.0),
)

_GRAVITY = 0.6
_JUMP_VELOCITY = 3.2
_RUN_SPEED = 1.2
_DASH_MULTIPLIER = 1.8
_MAX_X = 130.0
_PIT_Y = -8.0

# The boss stands in a doorway: the player must stop, fight, and only then can
# pass.  A boss that can be run past is not a speedrun obstacle, and a goal that
# sits past an unbreakable wall would make the world unwinnable.
_BOSS_X_RANGE = (92.0, 104.0)
_BOSS_HITS = 3
_WALL_X = 106.0
_GOAL_X = 108.0
_GOAL_HEIGHT = 14.0

MILESTONE_ORDER: tuple[str, ...] = ("start", "bridge", "ledge", "boss", "goalseal", "ending1")


class FakeSpeedrunEnv:
    """A tiny deterministic platformer with the SpeedrunEnv contract."""

    WIDTH = 160
    HEIGHT = 120
    FPS = 20

    def __init__(self, *, max_frames: int = 4_000, seed: int = 0, boss_hp: int = 12) -> None:
        self.action_space = ActionSpace.minimal()
        self.observation_shape = (self.HEIGHT, self.WIDTH, 3)
        self.max_frames = max_frames
        self.boss_hp_start = boss_hp if boss_hp != 12 else _BOSS_HITS
        self.seed = seed
        self._state: dict[str, Any] = {}
        self.reset(seed=seed)

    # ---------------------------------------------------------------- helpers
    def _new_state(self) -> dict[str, Any]:
        return {
            "frame": 0,
            "x": 0.0,
            "y": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "on_ground": True,
            "facing": 1,
            "boss_hp": self.boss_hp_start,
            "milestone": "start",
            "win": False,
            "dead": False,
            "attacks": 0,
        }

    @staticmethod
    def _ground_top(x: float) -> float:
        best = _PIT_Y - 1.0
        for x0, x1, top in _PLATFORMS:
            if x0 <= x <= x1:
                best = max(best, top)
        return best

    # ------------------------------------------------------------------- api
    def reset(self, *, seed: int | None = None, milestone: str | None = None) -> Obs:
        if seed is not None:
            self.seed = seed
        self._state = self._new_state()
        if milestone is not None:
            if milestone not in MILESTONE_ORDER:
                raise ValueError("unknown milestone %r" % (milestone,))
            self._state["milestone"] = milestone
            if MILESTONE_ORDER.index(milestone) >= MILESTONE_ORDER.index("boss"):
                self._state["x"] = 90.0
                self._state["y"] = 15.0
        return self._obs()

    def step(
        self, action: int, frames: int = 1, *, screenshot: bool = True, hold: bool = False
    ) -> StepResult:
        """Advance the toy world.

        `screenshot` and `hold` exist to match the real backends, where
        forcing a frame capture costs ~40 ms and where a held button needs a
        press/release edge.  The toy world has neither, so both are accepted
        and ignored - but they must be *accepted*, because the planner drives
        every backend through the same call.
        """
        reward = 0.0
        for _ in range(max(1, int(frames))):
            reward += self._tick(int(action))
            if self._state["win"] or self._state["dead"]:
                break
        obs = self._obs()
        done = bool(self._state["win"] or self._state["dead"])
        if int(self._state["frame"]) >= self.max_frames:
            done = True
        return StepResult(obs=obs, reward=reward, done=done, info={"milestone": obs.milestone})

    def snapshot(self) -> bytes:
        return json.dumps(self._state, sort_keys=True).encode("utf-8")

    def restore(self, snap: bytes) -> None:
        self._state = json.loads(bytes(snap).decode("utf-8"))

    def game_time_ms(self) -> int:
        return int(int(self._state["frame"]) * 1000 / self.FPS)

    def milestone(self) -> str | None:
        return self._state.get("milestone")

    def determinism_report(self) -> dict[str, Any]:
        return {
            "backend": "fake",
            "bit_exact_rollback": True,
            "single_frame_step": True,
            "seeded_rng": True,
            "notes": "toy world; used for pipeline regression tests only",
        }

    def close(self) -> None:
        self._state = {}

    # --------------------------------------------------------------- physics
    def _tick(self, action: int) -> float:
        s = self._state
        if s["win"] or s["dead"]:
            return 0.0
        s["frame"] = int(s["frame"]) + 1
        pressed = set(buttons_from_mask(action))

        if "left" in pressed:
            s["vx"] = -_RUN_SPEED
            s["facing"] = -1
        elif "right" in pressed:
            s["vx"] = _RUN_SPEED
            s["facing"] = 1
        else:
            s["vx"] = 0.0
        if "dash" in pressed:
            s["vx"] = float(s["vx"]) * _DASH_MULTIPLIER
        if "jump" in pressed and s["on_ground"]:
            s["vy"] = _JUMP_VELOCITY
            s["on_ground"] = False

        s["vy"] = float(s["vy"]) - _GRAVITY
        s["y"] = float(s["y"]) + float(s["vy"])
        new_x = float(np.clip(float(s["x"]) + float(s["vx"]), 0.0, _MAX_X))
        if int(s["boss_hp"]) > 0 and new_x > _WALL_X:
            # The doorway is shut until the boss is defeated.
            new_x = _WALL_X
        s["x"] = new_x

        top = self._ground_top(float(s["x"]))
        if float(s["y"]) <= top:
            s["y"] = top
            s["vy"] = 0.0
            s["on_ground"] = True
        else:
            s["on_ground"] = False
        if float(s["y"]) < _PIT_Y:
            s["dead"] = True
            return -1.0

        return self._progress(pressed)

    def _progress(self, pressed: set[str]) -> float:
        """Advance the milestone state machine and return the frame reward."""
        s = self._state
        reward = 0.0
        x = float(s["x"])
        current = s["milestone"]
        idx = MILESTONE_ORDER.index(current)

        # Progression is monotone in the route order, so each step is only
        # eligible once the previous one has been reached.  Without this the
        # later checks can never fire: reaching the bridge sets idx to bridge,
        # which makes the ledge condition (idx <= ledge) false even when the
        # player is standing on the ledge.
        if idx == MILESTONE_ORDER.index("start") and x > 41.0:
            reward += self._set_milestone("bridge", 1.0)
            idx = MILESTONE_ORDER.index("bridge")
        if idx == MILESTONE_ORDER.index("bridge") and x > 71.0 and float(s["y"]) >= 14.0:
            reward += self._set_milestone("ledge", 1.0)
            idx = MILESTONE_ORDER.index("ledge")
        if idx == MILESTONE_ORDER.index("ledge") and x > 88.0 and float(s["y"]) >= 14.0:
            reward += self._set_milestone("boss", 1.0)
            idx = MILESTONE_ORDER.index("boss")
        if s["milestone"] == "boss":
            low, high = _BOSS_X_RANGE
            if "attack" in pressed and low - 4.0 <= x <= high + 2.0:
                s["attacks"] = int(s["attacks"]) + 1
                s["boss_hp"] = max(0, int(s["boss_hp"]) - 1)
                reward += 0.1
            if int(s["boss_hp"]) <= 0:
                reward += self._set_milestone("goalseal", 5.0)
        if s["milestone"] == "goalseal" and x >= _GOAL_X and float(s["y"]) >= _GOAL_HEIGHT:
            reward += self._set_milestone("ending1", 10.0)
            s["win"] = True
        return reward

    def _set_milestone(self, name: str, reward: float) -> float:
        self._state["milestone"] = name
        return reward

    # ---------------------------------------------------------------- render
    def _obs(self) -> Obs:
        img = np.zeros(self.observation_shape, dtype=np.uint8)
        img[..., 2] = 45
        for x0, x1, top in _PLATFORMS:
            px0 = int(x0 / _MAX_X * self.WIDTH)
            px1 = int(x1 / _MAX_X * self.WIDTH)
            py = int(self.HEIGHT - 10 - top * 2.0)
            img[max(0, py) : max(0, py) + 3, px0:px1] = (120, 120, 120)
        px = int(float(self._state["x"]) / _MAX_X * self.WIDTH)
        py = int(self.HEIGHT - 12 - float(self._state["y"]) * 2.0)
        img[max(0, py - 5) : max(1, py), max(0, px - 3) : px + 3] = (240, 70, 70)
        if int(self._state["boss_hp"]) > 0:
            bx = int(98.0 / _MAX_X * self.WIDTH)
            by = int(self.HEIGHT - 12 - 15.0 * 2.0)
            img[max(0, by - 8) : max(1, by), max(0, bx - 5) : bx + 5] = (60, 220, 80)
        gx = int(_GOAL_X / _MAX_X * self.WIDTH)
        gy = int(self.HEIGHT - 12 - 15.0 * 2.0)
        img[max(0, gy - 10) : max(1, gy), max(0, gx - 2) : gx + 2] = (250, 230, 90)
        # The fake backend reports the same shape the CDP backend returns, so the
        # state framer and every consumer of it are exercised for real in tests.
        return Obs(
            rgb=img,
            state={
                "player": {
                    "x": round(float(self._state["x"]), 3),
                    "y": round(float(self._state["y"]), 3),
                    "vx": round(float(self._state["vx"]), 3),
                    "vy": round(float(self._state["vy"]), 3),
                    "on_ground": bool(self._state["on_ground"]),
                    "direction": int(self._state["facing"]),
                    "mapId": 1,
                },
                "frame": int(self._state["frame"]),
                "scene": "Scene_Map",
                "boss_hp": int(self._state["boss_hp"]),
                "attacks": int(self._state["attacks"]),
                "win": bool(self._state["win"]),
                "dead": bool(self._state["dead"]),
            },
            milestone=self._state["milestone"],
            frames=int(self._state["frame"]),
        )
