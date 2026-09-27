"""Project configuration loaded from config/*.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ash.utils.paths import PROJECT_ROOT, resolve_path

CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("%s must contain a mapping at the top level" % path)
    return data


@dataclass
class KeyBinding:
    """One logical button bound to one or more physical keys."""

    names: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    keycodes: list[int] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.names)

    def single(self) -> tuple[str, str, int]:
        if not self.names:
            raise ValueError("button has no key binding")
        code = self.codes[0] if self.codes else ""
        keycode = self.keycodes[0] if self.keycodes else 0
        return self.names[0], code, keycode


@dataclass
class GameConfig:
    name: str = "咸鱼喵喵 / Nyaruru Fishy Fight"
    steam_appid: int = 1478160
    speedrun_game: str = "w6j7mg7 6".replace(" ", "")
    speedrun_category: str = "n2yx001d"
    cdp_endpoint: str = "http://127.0.0.1:9222"
    cdp_port: int = 9222
    window: dict[str, Any] = field(default_factory=dict)
    fps: int = 60
    #: Seconds of game time between two consecutive agent observations/actions.
    #:
    #: This is THE single source of truth for the whole pipeline's time scale,
    #: and it exists because leaving it implicit broke everything silently: the
    #: agent stepped one game frame (1/60 s) while the corpus was sampled every
    #: 2 s, so the IDM learned which action explains a 1/60 s change and was then
    #: asked to label 2 s changes.  It answered with one constant class 99.8% of
    #: the time, and pi "converged" by predicting that class.  The paper uses
    #: 0.25 s for the environment timestep AND for the corpus, so both sides are
    #: derived from this number now (see control_frame_skip and corpus_fps).
    control_interval_s: float = 0.25
    #: The option the difficulty pick is answered with, as the game's own option
    #: text or a 1-based number; empty means refuse it (the round aborts there).
    #:
    #: The pick cannot be closed any other way - it has no cancel - so refusing it
    #: parks the game on that screen.  Easy is the default because the corpus is
    #: speedruns, which are all easy: training on easy runs while the agent plays
    #: another difficulty would compare two different games.  Per-difficulty
    #: training means changing this, not mixing them.
    difficulty_preset: str = ""
    keymap: dict[str, KeyBinding] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def control_frame_skip(self) -> int:
        """Game frames to advance per agent action (0.25 s * 60 fps = 15)."""
        return max(1, int(round(self.control_interval_s * self.fps)))

    @property
    def corpus_fps(self) -> float:
        """Corpus sampling rate that matches the agent's timestep (4 fps)."""
        return 1.0 / self.control_interval_s


def load_game_config(path: str | Path | None = None) -> GameConfig:
    data = _load_yaml(resolve_path(path) if path else CONFIG_DIR / "game.yaml")
    keymap: dict[str, KeyBinding] = {}
    for button, spec in (data.get("keymap") or {}).items():
        if isinstance(spec, list):
            keymap[button] = KeyBinding(names=[str(s) for s in spec])
        elif isinstance(spec, dict):
            keymap[button] = KeyBinding(
                names=[str(n) for n in spec.get("names", [])],
                codes=[str(c) for c in spec.get("codes", [])],
                keycodes=[int(k) for k in spec.get("keycodes", [])],
            )
        else:
            keymap[button] = KeyBinding(names=[str(spec)])
    return GameConfig(
        name=data.get("name", "咸鱼喵喵 / Nyaruru Fishy Fight"),
        steam_appid=int(data.get("steam_appid", 1478160)),
        speedrun_game=data.get("speedrun_game", "w6j7mg76"),
        speedrun_category=data.get("speedrun_category", "n2yx001d"),
        cdp_endpoint=data.get("cdp_endpoint", "http://127.0.0.1:9222"),
        cdp_port=int(data.get("cdp_port", 9222)),
        window=data.get("window", {}) or {},
        fps=int(data.get("fps", 60)),
        control_interval_s=float(data.get("control_interval_s", 0.25)),
        difficulty_preset=str(data.get("difficulty_preset", "") or "").strip(),
        keymap=keymap,
        raw=data,
    )
