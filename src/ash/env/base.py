"""Backend-agnostic environment contract for the speedrun agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ash.actions import ActionSpace


@dataclass
class Obs:
    """One observation of the game."""

    rgb: np.ndarray
    state: dict[str, Any] = field(default_factory=dict)
    milestone: str | None = None
    frames: int = 0

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("expected (H, W, 3) uint8 rgb, got %s" % (self.rgb.shape,))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.rgb.shape


@dataclass
class StepResult:
    obs: Obs
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SpeedrunEnv(Protocol):
    """What every backend must provide.

    The contract that makes search practical is the snapshot/restore pair: the
    caller must be able to fork the emulated game, explore, and then return to
    the exact same frame.  Backends that cannot guarantee bit-exactness must
    say so through determinism_report().
    """

    action_space: ActionSpace
    observation_shape: tuple[int, int, int]

    def reset(self, *, seed: int | None = None, milestone: str | None = None) -> Obs: ...

    def step(
        self, action: int, frames: int = 1, *, screenshot: bool = True, hold: bool = False
    ) -> StepResult: ...

    def snapshot(self) -> bytes: ...

    def restore(self, snap: bytes) -> None: ...

    def game_time_ms(self) -> int: ...

    def milestone(self) -> str | None: ...

    def determinism_report(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def make_env(backend: str = "auto", **kwargs: Any) -> SpeedrunEnv:
    """Construct a backend by name.

    'auto' picks CDP when a debugging endpoint is reachable, then the native
    X11 backend, and finally the scripted stand-in used by tests and CI.
    """
    backend = backend.lower()
    if backend in {"fake", "mock", "dummy"}:
        from ash.env.fake_backend import FakeSpeedrunEnv

        return FakeSpeedrunEnv(**kwargs)
    if backend == "x11":
        from ash.env.x11_backend import X11SpeedrunEnv

        return X11SpeedrunEnv(**kwargs)
    if backend == "cdp":
        from ash.env.cdp_backend import CdpSpeedrunEnv

        return CdpSpeedrunEnv(**kwargs)
    if backend == "wayland":
        from ash.env.wayland_backend import WaylandSpeedrunEnv

        return WaylandSpeedrunEnv(**kwargs)
    if backend != "auto":
        raise ValueError("unknown backend %r" % (backend,))

    from ash.env.cdp_backend import CdpSpeedrunEnv, cdp_endpoint_alive

    endpoint = kwargs.pop("endpoint", None)
    if cdp_endpoint_alive(endpoint):
        return CdpSpeedrunEnv(endpoint=endpoint, **kwargs)
    try:
        from ash.env.x11_backend import X11SpeedrunEnv

        return X11SpeedrunEnv(**kwargs)
    except Exception:
        from ash.env.fake_backend import FakeSpeedrunEnv

        return FakeSpeedrunEnv(**kwargs)
