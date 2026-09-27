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
    keymap: dict[str, KeyBinding] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


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
        keymap=keymap,
        raw=data,
    )


def load_milestones(path: str | Path | None = None) -> list[dict[str, Any]]:
    data = _load_yaml(resolve_path(path) if path else CONFIG_DIR / "milestones" / "ending1.yaml")
    return list(data.get("milestones", []))
