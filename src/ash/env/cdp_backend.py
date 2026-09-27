"""CDP backend: drive the game through its own embedded Chromium debugger.

Why this is the preferred backend:

  * the game loop can be stepped one update at a time, so a frame is a
    well-defined unit instead of a wall-clock guess;
  * the RPG Maker globals ($gamePlayer, $gameSwitches, ...) are readable, which
    gives the agent ground truth for milestone progress and for evaluation;
  * a snapshot of those globals plus the seeded RNG restores the game to the
    same frame, which is what makes search and deterministic replay possible.

Every capability is probed rather than assumed.  RPG Maker MZ runs inside an
nw.js/Chromium window whose graphics backend may or may not report frames fast
enough to keep up, and the desktop may be Wayland.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ash.actions import ActionSpace, buttons_from_mask, mask_from_buttons
from ash.capture.screencast import (
    CdpConnection,
    CdpError,
    Target,
    UnsafeSceneError,
    list_targets,
)
from ash.config import GameConfig, KeyBinding, load_game_config
from ash.env.base import Obs, StepResult
from ash.memory.js_source import js_source
from ash.utils.logging import get_logger

log = get_logger(__name__)

# Every virtual-time budget is in milliseconds; 16.7 ms approximates 60 Hz.
FRAME_MS = 1000.0 / 60.0


@dataclass
class CdpConfig:
    endpoint: str = "http://127.0.0.1:9222"
    keymap: dict[str, KeyBinding] = field(default_factory=dict)
    resize: tuple[int, int] | None = (128, 128)
    frame_ms: float = FRAME_MS
    settle_ms: float = 30.0


def cdp_endpoint_alive(endpoint: str | None = None) -> bool:
    from ash.capture.screencast import cdp_endpoint_alive as alive

    return alive(endpoint)


class CdpSpeedrunEnv:
    """SpeedrunEnv implemented on top of the Chrome DevTools Protocol."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        config: GameConfig | None = None,
        action_space: ActionSpace | None = None,
        resize: tuple[int, int] | None = (128, 128),
        install_pump: bool = True,
        seed: int = 1,
        target_index: int = 0,
        auto_connect: bool = True,
        enforce_safety: bool = True,
    ) -> None:
        #: Refuse to dispatch gameplay input outside Scene_Map.  Only the
        #: diagnostic key probes opt out, and they do so explicitly - the
        #: default has to be safe, because the failure it prevents (an
        #: untrained policy pressing its way into the player's save) is
        #: destructive and silent.
        self.enforce_safety = enforce_safety
        self.game_config = config or load_game_config()
        self.config = CdpConfig(
            endpoint=endpoint or self.game_config.cdp_endpoint,
            keymap=self.game_config.keymap,
            resize=resize,
        )
        self.action_space = action_space or ActionSpace.minimal()
        self.seed_value = seed
        self.observation_shape = (
            (resize[0] if resize else 0),
            (resize[1] if resize else 0),
            3,
        )
        self.conn: CdpConnection | None = None
        self.target: Target | None = None
        self._frame: np.ndarray | None = None
        self._frame_count = 0
        self._screencast_on = False
        self._frame_times: list[float] = []
        self._pump_installed = False
        self._rng_enabled = False
        self._milestone_resolver: Callable[[dict[str, Any]], str | None] | None = None
        self._install_pump = install_pump
        self._probe_cache: dict[str, Any] | None = None
        if auto_connect:
            self.connect(target_index=target_index)

    # ------------------------------------------------------------- lifecycle
    def connect(self, *, target_index: int = 0) -> CdpSpeedrunEnv:
        targets = list_targets(self.config.endpoint)
        if not targets:
            raise CdpError("no CDP targets at %s" % self.config.endpoint)
        pages = [t for t in targets if t.is_page] or targets
        if target_index >= len(pages):
            raise CdpError("target_index %d out of range (%d pages)" % (target_index, len(pages)))
        self.target = pages[target_index]
        if not self.target.web_socket_debugger_url:
            raise CdpError("target %s exposes no webSocketDebuggerUrl" % self.target.id)
        self.conn = CdpConnection(self.target.web_socket_debugger_url).open()
        self.conn.call("Runtime.enable")
        self.conn.call("Page.enable")
        if self._install_pump:
            self.install()
        return self

    def install(self, *, seed: int | None = None) -> dict[str, Any]:
        """Inject the in-page agent and start frame capture."""
        if self.conn is None:
            raise CdpError("not connected")
        # Also registered for future documents, so a page reload does not
        # silently lose the agent.
        self.conn.call("Page.addScriptToEvaluateOnNewDocument", {"source": js_source()})
        self.conn.evaluate(js_source())
        self._pump_installed = bool(
            self.conn.evaluate("!!(window.__ash && __ash.pump.install())")
        )
        self._rng_enabled = bool(
            self.conn.evaluate(
                "!!(window.__ash && __ash.rng.enable(%d))" % int(seed or self.seed_value)
            )
        )
        self.start_screencast()
        self._probe_cache = None
        return {
            "pump_installed": self._pump_installed,
            "rng_enabled": self._rng_enabled,
            "screencast": self._screencast_on,
        }

    def close(self, *, resume: bool = False) -> None:
        """Detach, leaving the game PAUSED unless resume is requested.

        Pausing on exit is deliberate.  When the agent stops, the character is
        standing wherever the run ended; this engine's enemies track the player
        relentlessly, so handing the loop back immediately means the character
        is beaten to death with nobody controlling it.  The safe default is to
        leave the ticker stopped.

        Pass resume=True only when a human is about to take over the keyboard.
        """
        try:
            if self.conn is not None:
                self.stop_screencast()
                if self._pump_installed:
                    try:
                        self.conn.evaluate(
                            "window.__ash && __ash.pump.uninstall(%s)"
                            % ("true" if resume else "false")
                        )
                    except CdpError:
                        pass
                self.conn.close()
        finally:
            self.conn = None

    def resume_game(self) -> bool:
        """Hand the game loop back to the engine and let the game run.

        The counterpart to the paused state close() leaves behind: use it when
        a human wants to keep playing the session the agent stopped in.
        """
        if self.conn is None:
            raise CdpError("not connected")
        try:
            raw = self.conn.evaluate(
                "JSON.stringify(window.__ash ? __ash.pump.resume()"
                " : {resumed:false, reason:'no __ash'})"
            )
            return bool(json.loads(raw).get("resumed")) if raw else False
        except (CdpError, ValueError):
            return False

    # ------------------------------------------------------------- screencast
    def start_screencast(self, *, quality: int = 80, max_width: int | None = None) -> bool:
        if self.conn is None:
            raise CdpError("not connected")
        params: dict[str, Any] = {"format": "jpeg", "quality": quality, "everyNthFrame": 1}
        if max_width:
            params["maxWidth"] = int(max_width)
        try:
            self.conn.call("Page.startScreencast", params)
        except CdpError as exc:
            log.warning("startScreencast unavailable: %s", exc)
            self._screencast_on = False
            return False
        self._screencast_on = True
        self._drain_frames(timeout=0.5)
        return True

    def stop_screencast(self) -> None:
        if self.conn is None or not self._screencast_on:
            return
        try:
            self.conn.call("Page.stopScreencast")
        except CdpError:
            pass
        self._screencast_on = False

    def _ack(self, session_id: Any) -> None:
        if session_id is None or self.conn is None:
            return
        try:
            self.conn.call("Page.screencastFrameAck", {"sessionId": session_id}, timeout=2.0)
        except CdpError:
            pass

    def _drain_frames(self, *, timeout: float = 0.2) -> int:
        """Collect screencastFrame events, keeping the newest frame."""
        if self.conn is None:
            return 0
        events = self.conn.drain_events(max_wait=timeout)
        got = 0
        for event in events:
            if event.get("method") != "Page.screencastFrame":
                continue
            params = event.get("params", {})
            data = params.get("data")
            if data:
                self._frame = _decode_jpeg_base64(data, self.config.resize)
                self._frame_count += 1
                got += 1
                self._frame_times.append(time.monotonic())
                if len(self._frame_times) > 128:
                    self._frame_times = self._frame_times[-128:]
            self._ack(params.get("sessionId"))
        return got

    def capture_once(self, *, timeout: float = 2.0) -> np.ndarray:
        """Force a fresh screenshot through Page.captureScreenshot.

        The screencast stream only emits on compositor damage; a paused game
        produces no new frames, so anything that needs the current picture after
        a step must use this call.
        """
        if self.conn is None:
            raise CdpError("not connected")
        result = self.conn.call(
            "Page.captureScreenshot",
            {"format": "jpeg", "quality": 80, "fromSurface": True, "captureBeyondViewport": False},
            timeout=timeout,
        )
        data = result.get("data")
        if not data:
            raise CdpError("captureScreenshot returned no data")
        self._frame = _decode_jpeg_base64(data, self.config.resize)
        self._frame_count += 1
        return self._frame

    # ------------------------------------------------------------------ input
    def _tap_binding(self, binding: Any) -> None:
        """Press and release one key across pumped frames.

        A trigger needs a release edge: pressing without releasing leaves the
        game's Input state latched, so the next press is not seen as a new
        trigger (Input.isStrictTriggered).  Every tap therefore brackets the
        keyDown with noop frames and always sends a keyUp.
        """
        self.step_frame(self.action_space.noop)
        self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "rawKeyDown"))
        self.step_frame(self.action_space.noop)
        self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "keyUp"))
        self.step_frame(self.action_space.noop)

    def _key_params(self, binding: KeyBinding, key_type: str) -> dict[str, Any]:
        name, code, keycode = binding.single()
        if not code:
            code = code_for_name(name)
        if not keycode:
            keycode = keycode_for_name(name)
        params: dict[str, Any] = {
            "type": key_type,
            "key": name,
            "code": code,
            "windowsVirtualKeyCode": keycode,
            "nativeVirtualKeyCode": keycode,
        }
        # "text" is only valid for keys that produce a character.  Sending it for
        # non-printable keys (arrows, Escape, Enter) is rejected by the protocol
        # with "Invalid 'text' parameter", and a single rejected key event aborts
        # the whole action.
        if key_type == "rawKeyDown" and name in TEXT_KEYS:
            params["text"] = name
            params["unmodifiedText"] = name
        return params

    def _bindings_for(self, action: int) -> list[Any]:
        if self.conn is None:
            raise CdpError("not connected")
        return [
            self.config.keymap[b]
            for b in buttons_from_mask(action)
            if self.config.keymap.get(b) and self.config.keymap[b].names
        ]

    def _check_safe(self) -> None:
        """Refuse to touch the keyboard outside Scene_Map gameplay.

        The runner asks unsafe_reason() before it acts, but this is the
        backstop: no caller, however written, gets to press gameplay keys on a
        title or menu screen, where those same keys mean "confirm".
        """
        if not self.enforce_safety:
            return
        reason = self.unsafe_reason()
        if reason:
            raise UnsafeSceneError("refusing to dispatch input: %s" % reason)

    def apply_action(self, action: int, *, settle_ms: float | None = None) -> None:
        """Press the buttons of an action mask, then release them.

        The release is unconditional: a game that never sees a keyup keeps a
        button held forever, which turns a single mistake into a broken episode.
        """
        self._check_safe()
        for binding in self._bindings_for(action):
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "rawKeyDown"))
        if settle_ms:
            time.sleep(settle_ms / 1000.0)
        for binding in self._bindings_for(action):
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "keyUp"))

    # --------------------------------------------------------------- stepping
    def step_frame(
        self, action: int, frames: int = 1, *, settle_ms: float | None = None, hold: bool = False
    ) -> int:
        """Apply an action and advance exactly frames game updates.

        With the frame pump installed the updates happen on demand; without it
        the only option is to sleep for the equivalent wall-clock time, which
        the determinism report records as inexact.

        Pump-mode ordering matters: the game samples Input on the frames it
        actually runs, so the key must be DOWN for the first pumped frame and
        released before a later pumped frame.  dispatch-both-then-pump (the
        previous order) let every pumped frame see an idle keyboard, which
        made trigger edges - dialogue advance, menu confirm - unreachable.

        hold=True keeps the key down across every pumped frame instead of
        releasing one frame early: variable-height jumps (hold longer = jump
        higher) and held movement need the key spanning the whole interval.
        """
        if self.conn is None:
            raise CdpError("not connected")
        self._check_safe()
        count = max(1, int(frames))
        if self._pump_installed:
            bindings = self._bindings_for(action)
            if bindings:
                for binding in bindings:
                    self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "rawKeyDown"))
                if hold:
                    ticks = self.conn.evaluate(
                        "window.__ash.pump.pump(%d)" % count, timeout=30.0
                    )
                else:
                    held = max(1, count - 1) if count > 1 else 1
                    ticks = self.conn.evaluate(
                        "window.__ash.pump.pump(%d)" % held, timeout=30.0
                    )
                for binding in bindings:
                    self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "keyUp"))
                if not hold and count > 1:
                    ticks2 = self.conn.evaluate(
                        "window.__ash.pump.pump(1)", timeout=30.0
                    )
                    ticks = (int(ticks or 0)) + int(ticks2 or 0)
            else:
                ticks = self.conn.evaluate(
                    "window.__ash.pump.pump(%d)" % count, timeout=30.0
                )
            return int(ticks or 0)
        self.apply_action(action, settle_ms=settle_ms or self.config.frame_ms)
        time.sleep(self.config.frame_ms * count / 1000.0)
        self._drain_frames(timeout=0.0)
        return count

    # ------------------------------------------------------------------- snap
    def snapshot(self) -> bytes:
        if self.conn is None:
            raise CdpError("not connected")
        payload = self.conn.evaluate("window.__ash.snapshot()", timeout=60.0)
        if not isinstance(payload, str):
            raise CdpError("snapshot returned %s" % type(payload).__name__)
        return payload.encode("utf-8")

    def restore(self, snap: bytes) -> None:
        if self.conn is None:
            raise CdpError("not connected")
        text = bytes(snap).decode("utf-8")
        count = self.conn.evaluate(
            "window.__ash.restore(%s)" % json.dumps(text), timeout=60.0
        )
        # -1 is the in-page "the snapshot belongs to another map" sentinel: a
        # rollout walked through a seam, so $dataMap now holds a different
        # map's data.  Reload the snapshot's map first, then retry.
        if isinstance(count, int) and count < 0:
            self._reload_snapshot_map(text)
            count = self.conn.evaluate(
                "window.__ash.restore(%s)" % json.dumps(text), timeout=60.0
            )
        if count is None or (isinstance(count, int) and count < 0):
            raise CdpError("restore did not report a captured-object count")
        self._report_event_graph_repair()

    def _report_event_graph_repair(self) -> None:
        """Surface a broken $gameMap._events caught by the in-page self-check.

        V.restore rebuilds any _events slot that is not a Game_Event (see
        V.repairEventGraph).  Measured live on map 6, the player object had been
        written into _events[13]; calcHibernate() copied it into _updateEvents,
        Game_Map.updateEventSync then threw on every frame, Game_Map.update
        aborted before the interpreter ran, and the cutscene froze mid-sequence
        (Nyaruru/error/ grew by ~16 files/s).  The graph is usable again after
        the repair, but a rollback that needed one was not a faithful rollback,
        so a search that keeps triggering this is running on a graph it cannot
        trust.
        """
        if self.conn is None:
            return
        try:
            raw = self.conn.evaluate(
                "JSON.stringify(window.__ash.lastEventGraphRepair || null)",
                timeout=10.0,
            )
            repaired = json.loads(raw) if raw else None
            if repaired:
                log.warning("restore repaired $gameMap._events: %s", repaired)
                self.conn.evaluate(
                    "window.__ash.lastEventGraphRepair = null", timeout=10.0
                )
        except (CdpError, ValueError):
            return

    def _reload_snapshot_map(self, snap_text: str) -> None:
        """Reload the map a snapshot belongs to, through the game's own path.

        A search rollout can walk through a seam and change the map.  The
        snapshot captures $gameMap but not $dataMap, so an in-place restore
        would leave the old map id/events/terrain next to the new map data -
        the tilemap renders the wrong layers (black background) and the player
        loses the ground.  `__ash.reloadMap` re-enters the map with
        reserveTransfer + SceneManager.goto(Scene_Map), which is exactly what
        an in-game transfer does, then this pumps frames until the map file has
        loaded and the scene has rebuilt for it.  The in-place restore is
        retried by the caller once the map matches again.
        """
        try:
            meta = json.loads(snap_text).get("__meta") or {}
        except (ValueError, AttributeError):
            return
        want = meta.get("mapId")
        if not isinstance(want, int) or want <= 0:
            raise CdpError("cannot reload map for snapshot without a mapId")
        x = meta.get("playerX")
        y = meta.get("playerY")
        x = int(x) if isinstance(x, (int, float)) else 0
        y = int(y) if isinstance(y, (int, float)) else 0
        self.conn.evaluate(
            "JSON.stringify((function(){try{return __ash.reloadMap(%d,%d,%d,2);}"
            "catch(e){return String(e).slice(0,200);}})())" % (want, x, y),
            timeout=20.0,
        )
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self._snapshot_map_ready(want):
                # The scene has rebuilt but has not drawn yet: the first
                # capture after onMapLoaded is pure black (measured unique=1).
                # Settle a few frames so the caller's observe() does not train
                # on an all-black frame.
                for _ in range(3):
                    self.step_frame(self.action_space.noop)
                return
            # The scene change and the async MapXXX.json fetch only progress
            # while the pump drives the game.
            self.step_frame(self.action_space.noop)
        raise CdpError("map reload to %d did not finish" % want)

    def _snapshot_map_ready(self, want: int) -> bool:
        """True once the scene is Scene_Map and `want`'s data is loaded."""
        if self.conn is None:
            return False
        try:
            raw = self.conn.evaluate(
                "JSON.stringify({scene:(window.SceneManager&&SceneManager._scene)"
                "?SceneManager._scene.constructor.name:null,"
                "map:(window.$gameMap?$gameMap._mapId:null),"
                "loaded:!!(window.DataManager&&DataManager.isMapLoaded"
                "&&DataManager.isMapLoaded())})",
                timeout=10.0,
            )
            state = json.loads(raw) if raw else {}
        except (CdpError, ValueError):
            return False
        return (
            state.get("scene") == "Scene_Map"
            and state.get("map") == want
            and bool(state.get("loaded"))
        )

    # -------------------------------------------------------------------- obs
    # ----------------------------------------------------------------- guard
    #: Scenes that mean a run is at risk.  Scene_Menu holds "back to town"
    #: (useful) next to "return to title / load game / exit game" (fatal),
    #: so a single stray keypress can end a run.  Scene_Gameover and the
    #: title are equally unwanted mid-run.
    DANGER_SCENES: tuple[str, ...] = (
        "Scene_Menu",
        "Scene_Gameover",
        "Scene_Title",
        "Scene_Load",
        "Scene_Save",
        "Scene_Options",
        "Scene_Shop",      # walking into a merchant stalls the run
        "Scene_Item",
        "Scene_Skill",
        "Scene_Status",
        "Scene_Equip",
        "Scene_SkillSt",   # NYA skill status screen (observed in a run)
        "Scene_Formation",
        "Scene_Debug",
        "Scene_Transport",  # teleport menu: harmless but stalls a walk
    )
    #: Scenes exited with Escape/cancel; the rest (Gameover/Title) need OK.
    ESCAPE_SCENES: tuple[str, ...] = (
        "Scene_Menu",
        "Scene_Shop",
        "Scene_Item",
        "Scene_Skill",
        "Scene_SkillSt",
        "Scene_Status",
        "Scene_Equip",
        "Scene_Formation",
        "Scene_Load",
        "Scene_Save",
        "Scene_Options",
        "Scene_Transport",
    )

    #: Run-integrity invariants: switch id -> required value.  A speedrun
    #: assumes normal difficulty and no challenge-mode selection; a random
    #: statue interaction can silently flip these and invalidate the route,
    #: so they are checked every decision rather than trusted.
    #: Authoritative difficulty mapping, read from
    #: Game_System.prototype.setDifficulty:
    #:
    #:   _difficulty ===  1 -> sw22 = true   (hard)
    #:   _difficulty === -1 -> sw43 = true   (easy)
    #:   _difficulty ===  0 -> both false    (normal)
    #:
    #: setDifficulty also calls adjustDifficultyHp(), which rescales current HP
    #: by the ratio of the new maximum - that is why the health cap differs per
    #: mode and why switching does not heal proportionally.
    #:
    #: User-reported facts that shape the guards below:
    #:   * the statue UI shows no indicator of the active mode, so neither the
    #:     screen nor the option colour can be trusted (the red text on "hard"
    #:     is a developer warning, not a selection marker);
    #:   * a hidden mode exists beyond the three statue buttons, so difficulty
    #:     must be verified rather than assumed.
    #: (Correction: an earlier note here claimed the hidden mode "nightmare"
    #: stays at _difficulty 0.  Checking the source showed "nightmare" only
    #: names skills and enemies there, so that claim was a guess and has been
    #: removed rather than left to mislead.)
    INVARIANTS: dict[int, bool] = {
        22: False,  # hard mode off  (changes enemy HP and contact damage)
        43: False,  # easy mode off  (this project runs on normal)
    }
    HARD_SWITCH = 22
    EASY_SWITCH = 43

    #: Difficulty value -> label.  -1 easy, 0 normal, 1 hard.  The hidden
    #: nightmare mode is not reachable through this field (it uses separate
    #: switches), so anything outside this table is reported as "other" rather
    #: than silently mapped to normal.
    DIFFICULTY_NAMES: dict[int, str] = {-1: "easy", 0: "normal", 1: "hard"}

    def difficulty(self) -> dict[str, Any]:
        """Read the authoritative difficulty from the game itself.

        The user confirmed two things that matter here:
          * the statue UI does NOT show which mode is active, so the screen
            cannot be used to tell; and
          * the red text on the hard option is a developer warning, not a
            currently-selected indicator.
        So the only trustworthy source is Game_System._difficulty, which is
        what this reads.
        """
        if self.conn is None:
            return {"difficulty": None, "name": "unknown"}
        # Two fields are involved, per the source:
        #   _difficulty          the LIVE difficulty (boss rush etc. may
        #                        change it temporarily)
        #   _difficultySettings  what the PLAYER chose; setDifficultySettings
        #                        writes this and then calls setDifficulty
        # A route-validity claim depends on the player's choice, so both are
        # reported and the settings value is the one to compare against.
        try:
            raw = self.conn.evaluate(
                "JSON.stringify({live: $gameSystem._difficulty,"
                " settings: $gameSystem._difficultySettings})"
            )
            data = json.loads(raw) if raw else {}
        except (CdpError, ValueError):
            data = {}
        live = data.get("live")
        settings = data.get("settings")
        name = self.DIFFICULTY_NAMES.get(
            settings, "other" if settings is not None else "unknown"
        )
        return {
            "difficulty": live,
            "settings": settings,
            "name": name,
            "temporarily_changed": (
                live is not None and settings is not None and live != settings
            ),
        }

    def check_invariants(self) -> dict[str, Any]:
        """Report any run-integrity switch that drifted from its required value."""
        if self.conn is None:
            return {"ok": True, "violations": []}
        try:
            raw = self.conn.evaluate(
                "JSON.stringify(window.$gameSwitches ? $gameSwitches._data : null)"
            )
            data = json.loads(raw) if raw else None
        except (CdpError, ValueError):
            return {"ok": True, "violations": []}
        if not data:
            return {"ok": True, "violations": []}
        violations = []
        for switch_id, required in self.INVARIANTS.items():
            actual = bool(data[switch_id]) if switch_id < len(data) else False
            if actual != required:
                violations.append(
                    {"switch": switch_id, "required": required, "actual": actual}
                )
        return {"ok": not violations, "violations": violations}

    def message_state(self) -> dict[str, Any]:
        """Is a dialogue message on screen, and does it ask a question?

        A plain message (no choices) still halts the interpreter and freezes
        the player until the confirm key is pressed, so a controller that only
        presses direction keys never gets control of a freshly started save -
        measured: the walker sat on the opening line of map 4 for 200
        decisions.  The caller confirms those lines with the game's own ok key
        (Z, which is also jump) and no-ops otherwise.  Choices are handled
        separately, because confirming one blindly can accept a difficulty
        prompt.
        """
        if self.conn is None:
            return {"busy": False, "choices": []}
        try:
            raw = self.conn.evaluate(
                "JSON.stringify((function(){"
                " var c = ($gameMessage.choices() || []);"
                " return {busy: $gameMessage.isBusy(), choices: c};"
                "})())"
            )
            return json.loads(raw) if raw else {"busy": False, "choices": []}
        except (CdpError, ValueError):
            return {"busy": False, "choices": []}

    def choice_state(self) -> dict[str, Any]:
        """Is the game showing a message choice, and what are the options?

        A blind confirm-mash is how a random walk selects "hard mode" on a
        difficulty statue and silently invalidates the whole route.  The
        caller uses this to back out of choices instead of accepting them.
        """
        if self.conn is None:
            return {"active": False, "choices": []}
        try:
            raw = self.conn.evaluate(
                "JSON.stringify((function(){"
                " var c = ($gameMessage.choices() || []);"
                " return {active: c.length > 0, choices: c,"
                "         index: $gameMessage._choiceIndex};"
                "})())"
            )
            return json.loads(raw) if raw else {"active": False, "choices": []}
        except (CdpError, ValueError):
            return {"active": False, "choices": []}

    def guard_choice(self, *, intent: str = "enter") -> dict[str, Any]:
        """Resolve a message choice according to the game's own layout.

        The user described the convention: with two options, the TOP one
        continues the dialogue and the BOTTOM one enters/travels.  So the
        right response depends on what the caller is trying to do:

          intent="continue" -> select the top option (index 0)
          intent="enter"    -> select the bottom option (last index)

        An earlier version of this method always pressed cancel, on the theory
        that any choice is a trap (a difficulty statue can invalidate a run).
        That was too blunt: it also blocked every legitimate "enter this
        area" prompt, so the runner could never take a branch.  The hazard is
        real but specific - it is the difficulty selection that must be
        avoided, not choices in general.
        """
        state = self.choice_state()
        if not state.get("active"):
            return {"choice": False, "selected": None}

        choices = list(state.get("choices") or [])

        # Safety first: a difficulty/mode prompt must not be accepted, or the
        # whole run's assumptions (enemy HP, damage) change silently.  This is
        # the one case where backing out beats choosing.
        danger = [c for c in choices if any(
            k in str(c) for k in ("困难", "难度", "Hard", "hard")
        )]
        if danger:
            cancel = self.config.keymap.get("cancel")
            report = {"choice": True, "choices": choices, "intent": intent,
                      "danger": danger, "selected": None, "moved": False}
            if cancel and cancel.names:
                try:
                    for _ in range(4):
                        self.step_frame(self.action_space.noop)
                        self.conn.call(
                            "Input.dispatchKeyEvent",
                            self._key_params(cancel, "rawKeyDown"),
                        )
                        self.step_frame(self.action_space.noop)
                        self.conn.call(
                            "Input.dispatchKeyEvent",
                            self._key_params(cancel, "keyUp"),
                        )
                        self.step_frame(self.action_space.noop)
                        if not self.choice_state().get("active"):
                            report["moved"] = True
                            break
                except CdpError as exc:
                    report["error"] = str(exc)[:120]
            return report

        if len(choices) == 1:
            index = 0
        elif intent == "continue":
            index = 0                      # top option = keep talking
        else:
            index = len(choices) - 1       # bottom option = enter/travel

        report: dict[str, Any] = {
            "choice": True,
            "choices": choices,
            "intent": intent,
            "selected": index,
            "selected_text": choices[index] if 0 <= index < len(choices) else None,
            "moved": False,
        }

        ok = self.config.keymap.get("jump") or self.config.keymap.get("interact")
        down = self.config.keymap.get("down")
        if not (ok and ok.names):
            return report

        def tap(binding: Any) -> None:
            self.step_frame(self.action_space.noop)
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "rawKeyDown"))
            self.step_frame(self.action_space.noop)
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "keyUp"))
            self.step_frame(self.action_space.noop)

        try:
            # Move the cursor from its current row to the wanted one, then
            # confirm.  The list is rendered top-down, so index 0 is the top.
            current = int(state.get("index") or 0)
            if down and down.names and index != current:
                steps = index - current
                key = down if steps > 0 else self.config.keymap.get("up")
                for _ in range(abs(steps)):
                    if key and key.names:
                        tap(key)
            tap(ok)
            report["moved"] = True
            report["resolved"] = not self.choice_state().get("active")
        except CdpError as exc:
            report["error"] = str(exc)[:120]
        return report

    def guard_scene(self) -> dict[str, Any]:
        """Detect and escape a run-threatening scene; returns a small report.

        Escaping is done by injecting Escape (menu) / confirm (title) rather
        than by touching game state, so the run stays honest: it is the same
        recovery a human would perform, just faster.
        """
        if self.conn is None:
            return {"scene": None, "escaped": False}
        try:
            scene = self.conn.evaluate(
                "SceneManager._scene ? SceneManager._scene.constructor.name : null"
            )
        except CdpError:
            return {"scene": None, "escaped": False}
        report = {"scene": scene, "escaped": False}
        if scene not in self.DANGER_SCENES:
            return report
        log.warning("danger scene %s detected; escaping", scene)
        # Scene_Gameover is NOT a single-OK screen, which is what an earlier
        # version of this guard assumed.  It is a two-button box
        # (Scene_Gameover.prototype.createSelectBox):
        #
        #     _selected = "up"    -> pressLoad()   (load / autoload a save)
        #     _selected = "down"  -> backToTown()  (respawn at town, full HP)
        #
        # and pressOk() *dispatches on the current selection* rather than
        # moving it.  Mashing OK therefore never got past the screen: with no
        # arrow press the selection stayed "up", pressLoad() saw the autosave
        # (DataManager.isSaveFileExist(0)) and tried to load it, and the scene
        # re-entered itself - the observed "Scene_Gameover" loop.
        #
        # The honest recovery is what the screen offers: pick "down" (back to
        # town, which also refills HP) and confirm once.
        if scene == "Scene_Gameover":
            ok = self.config.keymap.get("jump")
            if ok and ok.names:
                try:
                    # The select box only accepts input once the death
                    # animation has played out and onAlphaEasingCompleted has
                    # called _selectBox.active().  That takes a measured ~440
                    # game frames, so the budget here must be generous: an
                    # earlier 90-frame budget gave up long before the box was
                    # ready and reported "not escaped" on a screen that was
                    # simply still animating.
                    #
                    # Scene_Gameover.updateAnimationTimeline offers a skip:
                    # pressing the "menu" key (Escape) during the animation
                    # calls startFadeInBackSprite() immediately.  That is the
                    # game's own affordance, so it is used first, then the wait
                    # only has to cover the remaining fade.
                    menu = self.config.keymap.get("menu")
                    if menu and menu.names:
                        self._tap_binding(menu)
                    ready = False
                    for _ in range(240):          # 240 x 5 = 1200 frames
                        for _ in range(5):
                            self.step_frame(self.action_space.noop)
                        state = self.conn.evaluate(
                            "(function(){try{var s=SceneManager._scene;"
                            "if(!s||s.constructor.name!=='Scene_Gameover')"
                            "  return {scene:s?s.constructor.name:null};"
                            "var b=s._selectBox;"
                            "return {scene:'Scene_Gameover',"
                            "  box: b?!!b._active:false,"
                            "  selected: b?b._selected:null};}catch(e){"
                            "return {scene:null, err:String(e).slice(0,80)};}})()"
                        )
                        if not isinstance(state, dict):
                            continue
                        if state.get("scene") not in (None, "Scene_Gameover"):
                            report["escaped"] = True
                            report["scene"] = state["scene"]
                            return report
                        if state.get("box"):
                            ready = True
                            break
                    if not ready:
                        report["scene"] = "Scene_Gameover"
                        report["reason"] = "select box never became active"
                        return report
                    # Move the selection to "down" if it is not there already.
                    selected = self.conn.evaluate(
                        "(function(){try{return SceneManager._scene._selectBox._selected;}"
                        "catch(e){return null;}})()"
                    )
                    if selected != "down":
                        self._tap_binding(self.config.keymap.get("down"))
                    # Confirm, then give the scene switch time to happen.
                    self._tap_binding(ok)
                    for _ in range(120):
                        for _ in range(5):
                            self.step_frame(self.action_space.noop)
                        now = self.conn.evaluate(
                            "SceneManager._scene ? SceneManager._scene.constructor.name : null"
                        )
                        if now not in self.DANGER_SCENES:
                            report["escaped"] = True
                            report["scene"] = now
                            break
                    else:
                        report["scene"] = now
                except CdpError as exc:
                    report["error"] = str(exc)[:120]
            return report
        # Different NYA scenes close on different keys: the menu on Escape,
        # shops/items on the cancel key.  Try both rather than guessing.
        candidates = []
        for name in ("menu", "cancel"):
            binding = self.config.keymap.get(name)
            if binding and binding.names:
                candidates.append(binding)
        if scene in self.ESCAPE_SCENES and candidates:
            try:
                for _ in range(6):
                    for binding in candidates:
                        self.step_frame(self.action_space.noop)
                        self.conn.call(
                            "Input.dispatchKeyEvent",
                            self._key_params(binding, "rawKeyDown"),
                        )
                        self.step_frame(self.action_space.noop)
                        self.conn.call(
                            "Input.dispatchKeyEvent",
                            self._key_params(binding, "keyUp"),
                        )
                        self.step_frame(self.action_space.noop)
                        now = self.conn.evaluate(
                            "SceneManager._scene ? SceneManager._scene.constructor.name : null"
                        )
                        if now not in self.DANGER_SCENES:
                            report["escaped"] = True
                            report["scene"] = now
                            return report
            except CdpError as exc:
                report["error"] = str(exc)[:120]
        return report

    def observe(self, *, screenshot: bool = True) -> Obs:
        state = self.state()
        if screenshot or self._frame is None:
            try:
                self.capture_once()
            except CdpError as exc:
                # A stale cached frame is fine for scoring: the state readout
                # above is always fresh, and search only needs pixels at the
                # root for proposal ranking.
                log.warning("capture_once failed (%s); using cached frame", str(exc)[:120])
                if self._frame is None:
                    raise
        assert self._frame is not None
        return Obs(
            rgb=self._frame,
            state=state,
            milestone=self.milestone(),
            frames=self.game_frame_count(),
        )

    #: Combat entities are exposed through Game_Event.battleObject().
    #:
    #: Friend-or-foe is the engine's rule, not a sign test on `team`:
    #: Game_CharacterBase.checkSameTeam() treats -2 (neutral NPC) as friendly
    #: to everyone and -1 (free-for-all) as hostile to everyone, and compares
    #: the rest for equality.  The player's template is team 0, so 0 and -2
    #: are friendly and 1 / -1 are hostile.  An earlier readout used
    #: `team >= 0` for friendly, i.e. called every neutral NPC an enemy -
    #: which is how the walker attacked the shop keeper.
    #:
    #: battleObject._hp/_mhp are placeholder 1s for unlimitHp entities; the
    #: real pool is templateData().mhp.  unlimitHp:true means "not a killable
    #: enemy" (hazard/emitter).
    #:
    #: `targetable` mirrors the engine's own Game_Lily.isValidTarget
    #: (js/nya/nya_ai.js:370) so the reflex never swings at something the
    #: game would refuse to damage (an ally that only shows hit feedback, or
    #: an invincible / stagger-immune / special-hit entity).
    _ENEMY_EXPR = (
        "(function(){var out=[];var evs=$gameMap&&$gameMap.events?$gameMap.events():[];"
        "var pteam=0;"
        "try{pteam=($gamePlayer&&typeof $gamePlayer.battleTeam===\"function\")?$gamePlayer.battleTeam():0;}"
        "catch(pe){pteam=0;}"
        "for(var i=0;i<evs.length;i++){var e=evs[i];if(!e)continue;"
        "var bo=null;try{bo=(typeof e.battleObject===\"function\")?e.battleObject():null;}catch(err){bo=null;}"
        "if(!bo)continue;"
        "var td=null;try{td=(typeof bo.templateData===\"function\")?bo.templateData():null;}catch(t1){td=null;}"
        "var cur=(bo._hp!==undefined)?bo._hp:null;"
        "var mx=(bo.mhp!==undefined&&bo.mhp>1)?bo.mhp:null;"
        "var tmx=null;try{tmx=(td&&td.mhp!==undefined)?td.mhp:null;}catch(t2){tmx=null;}"
        "if(tmx!==null&&tmx>1){mx=tmx;}"
        "var team=0;try{team=(typeof e.battleTeam===\"function\")?e.battleTeam():((bo.team===undefined)?0:bo.team);}catch(t3){team=0;}"
        "var unlim=td?!!td.unlimitHp:false;"
        "var immue=false;try{immue=!!bo.immueStaggerAttack;}catch(t4){immue=false;}"
        "var special=0;try{special=bo.specailHitFlag|0;}catch(t5){special=0;}"
        "var dead=false;"
        "var inv=false;try{inv=(typeof e.isInvincible===\"function\")?!!e.isInvincible():false;}catch(t6){inv=false;}"
        "try{if(typeof e.isDeath===\"function\"){dead=!!e.isDeath();}}catch(d1){}"
        "var isMonster=false;try{isMonster=!!(td&&td.isMonster);}catch(t7){isMonster=false;}"
        "var hib=false;try{hib=(typeof e.shouldHibernate===\"function\")?!!e.shouldHibernate():false;}catch(err2){}"
        "var chasing=false;try{chasing=((e._targetId||0)>0);}catch(err3){}"
        "var img=String((bo.standingImage||''))+String((bo.movingImage||''));"
        "var mushroom=(img.indexOf('mushroom')>=0);"
        "var jumpStr=null;try{jumpStr=(bo.jumpStrength===undefined)?null:bo.jumpStrength;}catch(t8){jumpStr=null;}"
        "var moveStr=null;try{moveStr=(bo.moveStrength===undefined)?null:bo.moveStrength;}catch(t9){moveStr=null;}"
        "var friendly=(team===pteam)||(team===-2);"
        "out.push({id:e.eventId(),px:e.px,py:e.py,x:e.x,y:e.y,"
        " templateId:(typeof bo.templateId===\"function\")?bo.templateId():null,"
        " team:team,hp:cur,mhp:mx,unlimitHp:unlim,dead:dead,"
        " immueStaggerAttack:immue,specialHit:special,invincible:inv,isMonster:isMonster,"
        " mushroom:mushroom,jumpStrength:jumpStr,moveStrength:moveStr,"
        " statemType:(typeof e.statemType===\"function\")?e.statemType():null,"
        " friendly:friendly,hostile:!friendly,"
        " hibernate:hib,chasing:chasing,"
        " killable:(!friendly && !unlim && mx!==null && mx>1 && !dead),"
        " targetable:(!friendly && !unlim && !immue && special===0 && !dead && !inv),"
        " hazard:(!friendly && (unlim || cur===0))});}"
        "return out;})()"
    )
    @staticmethod
    def _check_js(expr: str, label: str) -> None:
        """Fail loudly on an unbalanced JS expression.

        Twice now a "//" comment inside a concatenated JS string commented out
        the rest of the snippet, leaving an unbalanced expression that only
        failed at runtime with "Unexpected end of input".  Checking at class
        definition time turns that into an immediate, obvious error.
        """
        if expr.count("(") != expr.count(")"):
            raise ValueError(
                "%s: unbalanced parentheses (%d open, %d close); a // comment"
                " inside a JS string literal comments out the remainder"
                % (label, expr.count("("), expr.count(")"))
            )
        if "//" in expr:
            raise ValueError(
                "%s: contains // which would comment out the rest of the JS"
                % label
            )

    _STATE_EXPR = (
        "JSON.stringify({player: window.__ash ? __ash.playerState() : null,"
        " scene: window.__ash ? __ash.sceneName() : null,"
        " frame: window.__ash ? __ash.frameCount() : null,"
        " item: window.__ash ? __ash.item() : null,"
        " switches: window.$gameSwitches ? $gameSwitches._data : null,"
        " variables: window.$gameVariables ? $gameVariables._data : null,"
        " physics: (window.$gamePlayer && $gamePlayer.px !== undefined) ?"
        " {px: $gamePlayer.px, py: $gamePlayer.py, vx: $gamePlayer.vx, vy: $gamePlayer.vy}"
        " : null,"
        " entities: " + _ENEMY_EXPR + ","
        " hp: (function(){try{var bo=($gamePlayer&&typeof $gamePlayer.battleObject===\"function\")"
        " ?$gamePlayer.battleObject():null; return bo?(bo._hp!==undefined?bo._hp:null):null;"
        " }catch(e){return null;}})(),"
        " maxHp: (function(){try{var bo=($gamePlayer&&typeof $gamePlayer.battleObject===\"function\")"
        " ?$gamePlayer.battleObject():null; return bo?(bo.mhp!==undefined?bo.mhp:null):null;"
        " }catch(e){return null;}})()})"
    )
    _check_js(_ENEMY_EXPR, "_ENEMY_EXPR")
    _check_js(_STATE_EXPR, "_STATE_EXPR")

    def state(self) -> dict[str, Any]:
        if self.conn is None:
            raise CdpError("not connected")
        raw = self.conn.evaluate(self._STATE_EXPR, timeout=10.0)
        if raw is None:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CdpError("state() got non-JSON: %s" % raw[:200]) from exc

    # ----------------------------------------------------------------- safety
    _SAFETY_EXPR = "window.__ash ? JSON.stringify(__ash.safety()) : null"
    _check_js(_SAFETY_EXPR, "_SAFETY_EXPR")

    def safety(self) -> dict[str, Any]:
        """In-page safety readout: current scene, and whether input is allowed.

        Returns {} when the agent is not installed, which callers must treat as
        unsafe rather than as "no objection".
        """
        if self.conn is None:
            raise CdpError("not connected")
        raw = self.conn.evaluate(self._SAFETY_EXPR, timeout=10.0)
        if raw is None:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CdpError("safety() got non-JSON: %s" % raw[:200]) from exc

    def unsafe_reason(self) -> str | None:
        """Why the keyboard must not be touched right now, or None if it may.

        The policy's verbs are gameplay verbs, but RPG Maker's ok/cancel keys
        are the same keys - so on a title or menu screen a "jump" press commits
        a menu selection, and with a save present that selection can load (or
        overwrite) it.  Anything that is not Scene_Map is unsafe, and so is an
        unreadable scene: failing closed is the point.
        """
        try:
            info = self.safety()
        except CdpError as exc:
            return "safety probe failed (%s)" % (str(exc)[:120],)
        if not info:
            return "safety probe unavailable: agent.js not installed"
        if not info.get("inGameplay"):
            return "scene %r is not Scene_Map gameplay" % (info.get("scene"),)
        return None

    def is_confirm_scene(self) -> bool:
        """True when the current scene is one a single "ok" press may dismiss.

        The allow-list lives in agent.js (V.CONFIRM_SCENES) so there is exactly
        one definition.  The hard gate stays in force for everything else.
        """
        try:
            return bool(self.safety().get("confirm"))
        except CdpError:
            return False

    def _confirm_mask(self) -> int:
        """The mask for the game's "ok" button, from the configured keymap."""
        names = self.game_config.keymap
        for name in ("interact", "jump"):
            binding = names.get(name)
            if binding and binding.names:
                return mask_from_buttons([name])
        raise CdpError("no ok button bound in the keymap (need interact or jump)")

    def press_ok(self) -> bool:
        """Press ONLY the ok button, and only on an allow-listed confirm scene.

        The scene gate exists because ok/cancel ARE the policy's jump/attack
        keys, so ok on a menu commits a selection - that is how a live run
        loaded the player's save.  But some non-gameplay scenes are input-gated
        rather than menu-like: the game's own transition screen waits for ok and
        never advances without it, so refusing all input does not protect
        anything, it just parks the game there permanently.

        This is the whole exception: one named button, an allow-listed scene,
        no random action, and the caller bounds how many times it may happen.
        """
        if not self.is_confirm_scene():
            raise UnsafeSceneError(
                "press_ok refused: scene %r is not in the confirm allow-list"
                % (self.safety().get("scene"),)
            )
        for binding in self._bindings_for(self._confirm_mask()):
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "rawKeyDown"))
        for binding in self._bindings_for(self._confirm_mask()):
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "keyUp"))
        return True

    def game_frame_count(self) -> int:
        if self.conn is None:
            return 0
        value = self.conn.evaluate("window.__ash ? __ash.frameCount() : null", timeout=10.0)
        return int(value or 0)

    def game_time_ms(self) -> int:
        """Wall-clock-independent elapsed game time derived from frame count."""
        frames = self.game_frame_count() or self._frame_count
        return int(frames * self.config.frame_ms)

    def milestone(self) -> str | None:
        if self._milestone_resolver is None:
            return None
        try:
            return self._milestone_resolver(self.state())
        except Exception as exc:  # pragma: no cover - resolver is user code
            log.debug("milestone resolver failed: %s", exc)
            return None

    def set_milestone_resolver(
        self, resolver: Callable[[dict[str, Any]], str | None] | None
    ) -> None:
        self._milestone_resolver = resolver

    # ---------------------------------------------------------------- probing
    def probe(self, *, with_rollback: bool = False) -> dict[str, Any]:
        """Report what this game instance actually supports.

        Nothing here is a hard requirement for the pipeline to run; the point is
        to replace assumptions with measurements before any training happens.

        Rollback measurement is OFF by default because it snapshots and
        restores live game objects: as part of the default probe it made every
        diagnostic run silently perform a restore, which crashed the game on a
        map with many events.
        """
        if self.conn is None:
            raise CdpError("not connected")
        report: dict[str, Any] = {
            "endpoint": self.config.endpoint,
            "target": {
                "id": self.target.id if self.target else None,
                "title": self.target.title if self.target else None,
                "url": self.target.url if self.target else None,
            },
        }
        report["version"] = _safe(self.conn.call, "Browser.getVersion")
        report["in_page"] = _safe(self.conn.evaluate, "window.__ash ? __ash.probe() : null")
        report["has_game_globals"] = bool(
            _safe(self.conn.evaluate, "!!(window.SceneManager && window.Graphics)")
        )
        report["pump"] = _safe(
            self.conn.evaluate,
            "window.__ash ? {installed: __ash.pump.installed, hijacked: __ash.pump.hijacked,"
            " ticks: __ash.pump.ticks} : null",
        )
        report["game_objects"] = _safe(
            self.conn.evaluate,
            "JSON.stringify(Object.keys(window).filter(function(k){"
            " return k.indexOf(String.fromCharCode(36)+String.fromCharCode(103)+\"ame\")===0;}))",
        )
        report["virtual_time"] = self._try_virtual_time()
        report["screencast"] = self._benchmark_screencast(samples=8)
        report["screenshot"] = self._benchmark_screenshot(samples=5)
        report["snapshot"] = self._benchmark_snapshot(samples=3)
        # Rollback testing is OPT-IN.  It takes a snapshot and restores it, which
        # mutates live game objects; running it unconditionally inside probe()
        # meant every diagnostic script silently performed a restore, and on a
        # map with many events that was enough to crash the game.  Callers that
        # actually want the measurement pass with_rollback=True (or call
        # rollback_consistency directly).
        if with_rollback:
            report["rollback"] = self.rollback_consistency(frames=8, trials=3)
        else:
            report["rollback"] = {"skipped": "pass with_rollback=True to measure"}
        report["document_hidden"] = _safe(self.conn.evaluate, "document.hidden")
        report["visibility_state"] = _safe(self.conn.evaluate, "document.visibilityState")
        self._probe_cache = report
        return report

    def _try_virtual_time(self) -> dict[str, Any]:
        assert self.conn is not None
        out: dict[str, Any] = {"supported": False}
        try:
            self.conn.call(
                "Emulation.setVirtualTimePolicy",
                {"policy": "pauseIfNetworkFetchesPending", "budget": 0},
                timeout=5.0,
            )
            out["pause_policy_accepted"] = True
            out["supported"] = True
        except CdpError as exc:
            out["error"] = str(exc)[:300]
        finally:
            try:
                self.conn.call("Emulation.setVirtualTimePolicy", {"policy": "advance"}, timeout=5.0)
            except CdpError:
                pass
        return out

    def _benchmark_screencast(self, *, samples: int = 8) -> dict[str, Any]:
        if not self._screencast_on:
            return {"available": False}
        start = time.monotonic()
        got = 0
        for _ in range(samples):
            got += self._drain_frames(timeout=0.25)
        elapsed = time.monotonic() - start
        intervals = [
            b - a
            for a, b in zip(self._frame_times, self._frame_times[1:], strict=False)
            if b > a
        ]
        return {
            "available": True,
            "frames_observed": got,
            "elapsed_s": round(elapsed, 3),
            "effective_fps": round(got / elapsed, 2) if elapsed > 0 else None,
            "median_interval_ms": round(1000 * median(intervals), 2) if intervals else None,
        }

    def _benchmark_screenshot(self, *, samples: int = 5) -> dict[str, Any]:
        times: list[float] = []
        for _ in range(samples):
            start = time.monotonic()
            try:
                self.capture_once(timeout=10.0)
            except CdpError as exc:
                return {"available": False, "error": str(exc)[:200]}
            times.append(time.monotonic() - start)
        return {
            "available": True,
            "median_ms": round(1000 * median(times), 2),
            "p95_ms": round(1000 * percentile(times, 0.95), 2),
        }

    def _benchmark_snapshot(self, *, samples: int = 3) -> dict[str, Any]:
        if self.conn is None or not self._pump_installed:
            return {"available": False}
        sizes: list[int] = []
        times: list[float] = []
        for _ in range(samples):
            start = time.monotonic()
            try:
                snap = self.snapshot()
            except CdpError as exc:
                return {"available": False, "error": str(exc)[:200]}
            times.append(time.monotonic() - start)
            sizes.append(len(snap))
        return {
            "available": True,
            "median_ms": round(1000 * median(times), 2),
            "bytes": int(median([float(s) for s in sizes])),
        }

    def rollback_consistency(self, *, frames: int = 8, trials: int = 3) -> dict[str, Any]:
        """Measure whether snapshot/restore reproduces the same frames.

        The comparison is on the serialised state hash and on the rendered frame
        hash, so a pass means both the simulation and the picture came back.  A
        failure here is the single most important finding Phase 0 can produce:
        without it there is no search, only imitation.
        """
        if self.conn is None:
            raise CdpError("not connected")
        if not self._pump_installed:
            return {"supported": False, "reason": "frame pump unavailable, rollback is untestable"}
        mask = 1 << 3  # the jump button: guaranteed to change the simulation
        # Warm up before measuring.  The very first step_frame() after the pump
        # takes over advances one frame FEWER than requested (measured 7 for a
        # request of 8; every later call is exact).  Measuring immediately made
        # trial 0 compare a 7-frame run against an 8-frame run and report
        # state_match_rate 0.667 - a startup artefact presented as a
        # determinism failure, on top of the snapshot bug that made the whole
        # check meaningless.  One throwaway step removes it.
        self.step_frame(self.action_space.noop)
        trials_out: list[dict[str, Any]] = []
        for _ in range(trials):
            snap = self.snapshot()
            fc0 = self.game_frame_count()
            self.step_frame(mask, frames=frames)
            advanced_a = self.game_frame_count() - fc0
            state_a = self.conn.evaluate("__ash.hashState()")
            frame_a = self.capture_once().copy()
            self.restore(snap)
            self.step_frame(mask, frames=frames)
            advanced_b = self.game_frame_count() - fc0
            state_b = self.conn.evaluate("__ash.hashState()")
            frame_b = self.capture_once()
            # Pixel differences are reported by magnitude, not only as a
            # boolean: the simulation is exact while a handful of pixels can
            # still differ by one unit from render-side phase, and calling
            # that "not reproducible" hides the distinction.
            diff = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))
            trials_out.append({
                "state_match": state_a == state_b,
                "frame_match": bool(np.array_equal(frame_a, frame_b)),
                "frames_advanced": (advanced_a, advanced_b),
                "pixels_changed": int((diff > 0).sum()),
                "pixels_changed_fraction": round(float((diff > 0).mean()), 6),
                "mean_abs_diff": round(float(diff.mean()), 4),
                "rng_enabled": self._rng_enabled,
            })
        state_matches = sum(1 for t in trials_out if t["state_match"])
        frame_matches = sum(1 for t in trials_out if t["frame_match"])
        return {
            "supported": True,
            "trials": trials_out,
            "state_match_rate": state_matches / len(trials_out),
            "frame_match_rate": frame_matches / len(trials_out),
            "verdict": _rollback_verdict(state_matches, frame_matches, len(trials_out)),
        }

    def determinism_report(self) -> dict[str, Any]:
        cached = self._probe_cache or {}
        rollback = cached.get("rollback") or {}
        return {
            "backend": "cdp",
            "bit_exact_rollback": rollback.get("verdict") == "bit_exact",
            "rollback_available": bool(rollback.get("supported", False)),
            "single_frame_step": self._pump_installed,
            "seeded_rng": self._rng_enabled,
            "screencast": cached.get("screencast"),
            "screenshot": cached.get("screenshot"),
            "virtual_time": cached.get("virtual_time"),
            "notes": "call probe() before relying on this report",
        }

    # ------------------------------------------------------- SpeedrunEnv api
    def reset(self, *, seed: int | None = None, milestone: str | None = None) -> Obs:
        if self.conn is None:
            raise CdpError("not connected")
        if seed is not None and seed != self.seed_value:
            self.seed_value = seed
            self.conn.evaluate("__ash.rng.setSeed(%d)" % int(seed))
        return self.observe(screenshot=True)

    def step(
        self, action: int, frames: int = 1, *, screenshot: bool = True, hold: bool = False
    ) -> StepResult:
        """Advance one step; screenshot=False skips the ~40ms forced capture.

        The observation then carries the real state/milestone readout but a
        possibly stale image. Search rollouts that score by state (the
        milestone potential) pass False and pay for pixels only at the
        root, where the proposal policy actually looks.
        """
        self.step_frame(action, frames=frames, hold=hold)
        obs = self.observe(screenshot=screenshot)
        return StepResult(obs=obs, reward=0.0, done=False, info={})


def _safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a probe step, turning a hard failure into a recorded string."""
    try:
        return fn(*args, **kwargs)
    except CdpError as exc:
        return {"error": str(exc)[:300]}
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}


def _decode_jpeg_base64(data: str, resize: tuple[int, int] | None) -> np.ndarray:
    import cv2

    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise CdpError("could not decode a JPEG frame")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize is not None and img.shape[:2] != tuple(resize):
        img = cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[idx])


def _rollback_verdict(state_matches: int, frame_matches: int, total: int) -> str:
    if total == 0:
        return "untested"
    if state_matches == total and frame_matches == total:
        return "bit_exact"
    if state_matches == total:
        return "state_exact_frames_differ"
    if state_matches >= total - 1:
        return "mostly_reproducible"
    return "not_reproducible"


# Keyboard data for the keys a 2-D platformer needs.  Chrome is happy with just
# windowsVirtualKeyCode for many games, but RPG Maker MZ reads KeyboardEvent
# .key / .code, so the mapping is explicit rather than guessed at run time.
_KEY_TABLE: dict[str, tuple[str, str, int]] = {
    "ArrowLeft": ("ArrowLeft", "ArrowLeft", 37),
    "ArrowUp": ("ArrowUp", "ArrowUp", 38),
    "ArrowRight": ("ArrowRight", "ArrowRight", 39),
    "ArrowDown": ("ArrowDown", "ArrowDown", 40),
    " ": ("Space", "Space", 32),
    "Enter": ("Enter", "Enter", 13),
    "Escape": ("Escape", "Escape", 27),
    "Shift": ("Shift", "ShiftLeft", 16),
    "Control": ("Control", "ControlLeft", 17),
    "Tab": ("Tab", "Tab", 9),
    "Backspace": ("Backspace", "Backspace", 8),
    ",": (",", "Comma", 188),
    ".": (".", "Period", 190),
    "/": ("/", "Slash", 191),
    ";": (";", "Semicolon", 186),
    "[": ("[", "BracketLeft", 219),
    "]": ("]", "BracketRight", 221),
    "-": ("-", "Minus", 189),
    "=": ("=", "Equal", 187),
}

# Keys that produce a text character: those are the only ones allowed to carry
# the "text"/"unmodifiedText" fields in Input.dispatchKeyEvent.  Non-printable
# keys (arrows, Escape, ...) must omit them or the protocol rejects the event.
TEXT_KEYS: set[str] = {
    " ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".",
    "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=",
    ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[",
    "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i",
    "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
    "y", "z", "{", "|", "}", "~",
}


def code_for_name(name: str) -> str:
    if name in _KEY_TABLE:
        return _KEY_TABLE[name][1]
    if len(name) == 1 and name.isalpha():
        return "Key" + name.upper()
    if len(name) == 1 and name.isdigit():
        return "Digit" + name
    return name


def keycode_for_name(name: str) -> int:
    if name in _KEY_TABLE:
        return _KEY_TABLE[name][2]
    if len(name) == 1:
        upper = name.upper()
        if "A" <= upper <= "Z":
            return ord(upper)
        if name.isdigit():
            return ord(name)
    return 0