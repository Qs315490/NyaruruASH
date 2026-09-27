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
        drive: str = "pump",
        difficulty: str | int | None = None,
    ) -> None:
        #: The option the operator preset for the difficulty pick, as an index,
        #: a 1-based number, or a substring of the option text.  None refuses the
        #: pick (the round aborts there) - answering it is the operator's call,
        #: never this layer's, so it only ever happens when asked for.
        self.difficulty_option = difficulty
        #: How frames get advanced.
        #:
        #: "pump"     - the agent takes the ticker and advances frames by hand.
        #:              Required for deterministic replay/search, but it CHANGES
        #:              the game: with a synthetic ticker clock this game's hurt
        #:              state never exits (measured: 1200 pumped frames pinned at
        #:              _pRealState 6, and a trap that teleports correctly under
        #:              the engine's own loop never resolved).
        #: "realtime" - the engine keeps its loop; the agent only dispatches
        #:              input on a real-time schedule.  Self-play must use this.
        if drive not in ("pump", "realtime"):
            raise ValueError("drive must be 'pump' or 'realtime', got %r" % (drive,))
        self.drive = drive
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
        #: Opt-in per-step screencast logging (see `record_step_frames`).  The
        #: stream is normally coalesced to its newest frame; the forward-window
        #: work needs the frames INSIDE a 0.25 s hold, which is the 30 fps
        #: resolution the video side has and this side otherwise throws away.
        self._recording_steps = False
        self._step_frames: list[list[tuple[float, np.ndarray]]] = []
        self._current_step: list[tuple[float, np.ndarray]] | None = None
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
        """Inject the in-page agent; hand-drive frames only in pump mode."""
        if self.conn is None:
            raise CdpError("not connected")
        # Also registered for future documents, so a page reload does not
        # silently lose the agent.
        self.conn.call("Page.addScriptToEvaluateOnNewDocument", {"source": js_source()})
        self.conn.evaluate(js_source())
        if self.drive == "realtime":
            # Leave the engine's own loop alone.  Measured on this game: a
            # hand-driven ticker pins the hurt state (_pRealState 6) forever -
            # 1200 pumped frames never left it - so the character cannot move
            # and a damage trap never resolves, while the same trap teleports
            # correctly when the engine drives itself.  Self-play needs the
            # game to behave normally far more than it needs determinism.
            # Start the engine's loop, not merely "stop owning it".  Calling
            # pump.uninstall(true) here is NOT enough: it early-returns when the
            # freshly injected agent has no pump installed, which is exactly the
            # case after a previous run left the ticker stopped - the round then
            # ran a whole 40 steps against a frozen game and nothing the agent
            # did had any effect.  pump.resume() releases a pump if one exists
            # and starts the ticker either way.
            resumed = self.conn.evaluate(
                "JSON.stringify(window.__ash ? __ash.pump.resume()"
                " : {resumed:false, reason:'no __ash'})"
            )
            self._pump_installed = False
            self._ticker_resumed = bool(json.loads(resumed).get("resumed")) if resumed else False
        else:
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
                # Lift any held button first.  Leaving the game paused with a
                # movement key still down would let the character walk off a
                # ledge the moment a human resumes the ticker.
                self.release()
                self.stop_screencast()
                if self._pump_installed:
                    try:
                        self.conn.evaluate(
                            "window.__ash && __ash.pump.uninstall(%s)"
                            % ("true" if resume else "false")
                        )
                    except CdpError:
                        pass
                elif self.drive == "realtime" and not resume:
                    # No pump to uninstall.  The paused exit is deliberate - an
                    # unattended character is beaten to death - and it must not
                    # silently become "keep playing" just because the drive mode
                    # changed.
                    #
                    # `ticker.stop()` alone is NOT a pause: measured, the ticker
                    # came back about two seconds later because the engine calls
                    # `Graphics._app.start()` on its own, so a stopped agent left
                    # the character in a monster area being killed.  `pauseGame`
                    # also shadows the ticker's start() while the agent holds the
                    # loop, which is what makes it stay stopped.
                    self._pause_ticker_verified()
                self.conn.close()
        finally:
            self.conn = None

    #: Reads the engine ticker without the agent, so a status check does not
    #: itself start it (installing the agent in realtime mode starts the loop).
    _TICKER_EXPR = (
        "(function(){try{var t=window.Graphics&&Graphics.app&&Graphics.app.ticker;"
        "return JSON.stringify({started:!!(t&&t.started),"
        " guarded:!!(t&&t.__ashGuarded),"
        " frames:(window.Graphics&&Graphics.frameCount)||null});}catch(e){"
        "return null;}})()"
    )

    def ticker_state(self) -> dict[str, Any]:
        """The engine loop's state, readable with or without the agent."""
        if self.conn is None:
            return {}
        raw = self.conn.evaluate(self._TICKER_EXPR)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError:
            return {}

    def is_paused(self, *, settle_s: float = 0.8) -> bool:
        """True when the loop is stopped AND stays stopped.

        One read right after stop() is not evidence: the engine restarts the
        ticker by itself, and a single sample cannot tell "paused" from "about
        to resume".  Waiting and reading again is the difference."""
        if self.ticker_state().get("started"):
            return False
        time.sleep(max(0.0, settle_s))
        return not self.ticker_state().get("started")

    def pause_game(self) -> dict[str, Any]:
        """Stop the loop and hold it stopped.  Verified, never assumed.

        Returns {"paused", "guarded", "reason"}.  Uses `__ash.pauseGame()` when
        the agent is installed (it shadows the ticker's start(), without which
        the engine resumes the game on its own - measured, ~2 s), and falls back
        to a plain stop for a page where the agent is not injected."""
        out: dict[str, Any] = {"paused": False, "guarded": False, "reason": None}
        if self.conn is None:
            out["reason"] = "not connected"
            return out
        try:
            self.release()
            raw = self.conn.evaluate(
                "(window.__ash&&__ash.pauseGame)?JSON.stringify(__ash.pauseGame()):"
                "((window.Graphics&&Graphics.app&&Graphics.app.ticker)?"
                "(Graphics.app.ticker.stop(),"
                "JSON.stringify({paused:true,guarded:false,fallback:true})):null)"
            )
            out.update(json.loads(raw) if raw else {})
        except (CdpError, ValueError) as exc:
            out["reason"] = str(exc)[:120]
            return out
        if not self.is_paused():
            out["paused"] = False
            out["reason"] = out.get("reason") or "the ticker restarted after being stopped"
            log.warning("pause did not hold: %s", out["reason"])
        return out

    def _pause_ticker_verified(self) -> None:
        out = self.pause_game()
        if not out.get("paused"):
            log.warning("game may still be running after close(): %s", out.get("reason"))

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
                frame = _decode_jpeg_base64(data, self.config.resize)
                self._frame = frame
                self._frame_count += 1
                got += 1
                self._frame_times.append(time.monotonic())
                if len(self._frame_times) > 128:
                    self._frame_times = self._frame_times[-128:]
                self._record_step_frame(frame, params.get("metadata") or {})
            self._ack(params.get("sessionId"))
        return got

    def _record_step_frame(self, frame: np.ndarray, metadata: dict[str, Any]) -> None:
        """Append a frame to the current step's log, if one is being recorded."""
        if self._current_step is None:
            return
        stamp = metadata.get("timestamp")
        self._current_step.append(
            (float(stamp) if isinstance(stamp, (int, float)) else time.monotonic(), frame))

    def record_step_frames(self, on: bool = True) -> None:
        """Start (or stop) keeping every screencast frame that arrives during a step.

        Off by default and harmless when off: the stream keeps being coalesced
        to its newest frame exactly as before.  On, each realtime hold is logged
        as its own frame sequence - the 0.25 s the buttons were down, sampled at
        whatever rate the compositor produces.  `take_step_frames()` returns and
        clears the log; entries line up one per dispatched action, so gate
        handling that skipped a step simply has no entry.
        """
        self._recording_steps = bool(on)
        self._step_frames = []
        if not on:
            self._current_step = None

    def take_step_frames(self) -> list[list[tuple[float, np.ndarray]]]:
        """Return and clear the recorded per-step frame sequences."""
        out, self._step_frames = self._step_frames, []
        return out

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

    def _check_safe(self, *, escape: bool = False, difficulty: bool = False,
                    gameover: bool = False) -> None:
        """Refuse to touch the keyboard outside Scene_Map gameplay.

        The runner asks unsafe_reason() before it acts, but this is the
        backstop: no caller, however written, gets to press gameplay keys on a
        title or menu screen, where those same keys mean "confirm".

        `escape=True` and `difficulty=True` are the two narrow modes, and neither
        is a bypass: the gate is still the gate.  `escape` permits exactly one
        thing - the cancel button, on a scene agent.js lists as a menu, with no
        choice pending (menu_escape_reason).  `difficulty` permits navigating and
        confirming exactly one thing - the difficulty pick the operator preset -
        and refuses any choice that is not that one (difficulty_reason), so it can
        never become a way to answer arbitrary dialogue.  Everything else still
        fails here, and enforce_safety=False still switches the whole gate off for
        diagnostic probes.
        """
        if not self.enforce_safety:
            return
        if escape:
            reason = self.menu_escape_reason()
        elif difficulty:
            reason = self.difficulty_reason()
        elif gameover:
            reason = self.gameover_reason()
        else:
            reason = self.unsafe_reason()
        if reason:
            raise UnsafeSceneError("refusing to dispatch input: %s" % reason)

    def gameover_reason(self) -> str | None:
        """Verdict for the narrow GAME OVER mode: only a save load may be pressed.

        Anything other than "on GAME OVER with the save-load entry highlighted" is
        refused, so this mode can never become a way to press ok on an
        unrecognised screen.
        """
        try:
            info = self.safety() or {}
        except CdpError as exc:
            return "safety probe failed (%s)" % (str(exc)[:120],)
        if info.get("scene") != "Scene_Gameover":
            return "not on the game over screen (%s)" % info.get("scene")
        for entry in info.get("menuEntries") or []:
            if entry.get("panel") != "_selectBox":
                continue
            if entry.get("name") in (self.GAMEOVER_SAFE_ENTRY, self.GAMEOVER_UNSAFE_ENTRY):
                return None
            return "unrecognised entry %r" % (entry.get("name"),)
        return "the highlighted entry could not be read"

    def menu_escape_reason(self) -> str | None:
        """None when a bounded cancel-only menu escape may be attempted.

        Narrower than every other input path in the class: a named scene
        (V.MENU_SCENES, in agent.js), a named button (cancel), no directions, no
        ok, and a cap imposed by the caller.  It exists because an agent that
        walks into a menu could otherwise never leave it - gameplay input is
        refused there for good reason - so a single accidental press cost the
        whole round and needed a human every time.
        """
        try:
            info = self.safety()
        except CdpError as exc:
            return "safety probe failed (%s)" % (str(exc)[:120],)
        if not info:
            return "safety probe unavailable: agent.js not installed"
        if info.get("inGameplay"):
            return "already in gameplay"
        if info.get("awaitingChoice"):
            return "a dialogue choice is awaiting an answer: cancel would answer it"
        if info.get("messageBusy"):
            return "a message is waiting: only ok advances it, and ok is not allowed"
        if not info.get("menu"):
            return "scene %r is not a known menu template" % (info.get("scene"),)
        if self.drive == "realtime" and not info.get("tickerRunning"):
            return "the engine's ticker is not running, so cancel would go nowhere"
        return None

    def _cancel_mask(self) -> int:
        """The mask for the game's cancel button, from the configured keymap."""
        binding = self.game_config.keymap.get("cancel")
        if binding is None or not binding.names:
            raise CdpError("no cancel button bound in the keymap")
        return mask_from_buttons(["cancel"])

    def _dispatch(self, action: int) -> None:
        """Tap the buttons of a mask: press, brief settle, release."""
        bindings = self._bindings_for(action)
        for binding in bindings:
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "rawKeyDown"))
        time.sleep(self.config.frame_ms / 1000.0)
        for binding in bindings:
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "keyUp"))

    def _poll_safety(self, predicate, *, tries: int = 8) -> dict:
        """Read safety until `predicate` holds, bounded by `tries` frames.

        A scene change is not visible on the frame the key is released: MZ runs
        the transition over the next few frames.  Reading once and concluding
        "it did not work" is what made the menu escape report failure while the
        game was in fact leaving - verified by the scene being Scene_Map again
        moments after the escape gave up.
        """
        info = self.safety() or {}
        for _ in range(max(0, int(tries) - 1)):
            if predicate(info):
                return info
            time.sleep(self.config.frame_ms / 1000.0)
            info = self.safety() or {}
        return info

    def _between_presses(self) -> None:
        """Hold the key UP for at least one sampled frame before pressing again.

        The game reads its input once per 60 Hz frame and MZ triggers on a
        transition, so a release shorter than a frame is never seen: four cancels
        dispatched back to back registered as at most one.  That is why the agent
        sat in Scene_Shop after "four" cancel presses while a human exits it with
        one - the presses were being sent, not observed.
        """
        # Two frames: one to sample the release, one of margin for the round trip
        # that reads the scene back in between.
        time.sleep(2.0 * self.config.frame_ms / 1000.0)

    def difficulty_choice_pending(self) -> bool:
        """True when the awaiting choice is the guarded (difficulty) pick."""
        try:
            info = self.safety() or {}
        except CdpError:
            return False
        return bool(info.get("awaitingChoice") and info.get("guardedChoice"))

    #: The one entry on the GAME OVER screen that may be committed.  The game's
    #: own symbols say which is which, and the guardrail refuses the other; an
    #: automatic routine that used to live here selected the refused one and had
    #: no callers, so it both contradicted the policy and never ran.
    GAMEOVER_SAFE_ENTRY = "STATIC_TEXT_CONTINUE_YES"
    GAMEOVER_UNSAFE_ENTRY = "STATIC_TEXT_CONTINUE_NO"

    def gameover_recovery_pending(self) -> bool:
        """True on GAME OVER with the cursor somewhere other than the save load.

        The screen is a two-button box and only the save-load entry may be
        committed.  When the cursor sits on the other one the gate refuses input
        and, before this existed, the whole round was thrown away - measured, a
        2500-step collection died after 1314 steps on exactly that.
        """
        try:
            info = self.safety() or {}
        except CdpError:
            return False
        if info.get("scene") != "Scene_Gameover":
            return False
        for entry in info.get("menuEntries") or []:
            if entry.get("panel") != "_selectBox":
                continue
            name = entry.get("name")
            if name == self.GAMEOVER_SAFE_ENTRY:
                return False                 # already safe: the gate allows it
            if name == self.GAMEOVER_UNSAFE_ENTRY:
                return True
        return False

    def resolve_gameover(self, *, max_presses: int = 12) -> dict[str, Any]:
        """Move the GAME OVER cursor onto the save load and commit it.

        Returns {"resolved", "presses", "reason"}.  Same contract as
        `resolve_difficulty`: it answers nothing, it only *navigates to* the one
        entry the policy already allows, the presses are bounded, and success is
        verified by re-reading the scene instead of assumed - "I pressed up then
        ok" is not evidence that the game left the screen.

        Measured on the real game: a death with the cursor on the refused entry
        resolved in SIX presses, i.e. it used the whole budget of 6 - the load
        needs a few confirms to take effect, so the bound has margin now (12)
        rather than being one unlucky frame away from giving up.

        The presses go through `_dispatch`, not `step_frame`: `step_frame` runs
        the ordinary gate, which refuses this screen by design, and refusing to
        move off the entry the policy rejects would leave the round dead-ended.
        The narrow `gameover` verdict is checked before each press instead.
        """
        out: dict[str, Any] = {"resolved": False, "presses": 0, "reason": None}
        try:
            info = self.safety() or {}
            up = mask_from_buttons(["up"])
            ok = self._confirm_mask()
        except CdpError as exc:
            out["reason"] = str(exc)[:160]
            return out
        if info.get("scene") != "Scene_Gameover":
            # Checked before the entries so the reason names the real problem: a
            # Scene_Map payload has no select box at all, and reporting "the
            # highlighted entry could not be read" sends the reader hunting for a
            # cursor that was never there.
            out["reason"] = "not on the game over screen (%s)" % info.get("scene")
            return out
        selected = None
        for entry in info.get("menuEntries") or []:
            if entry.get("panel") == "_selectBox":
                selected = entry.get("name")
        if selected == self.GAMEOVER_SAFE_ENTRY:
            out["reason"] = "the save-load entry is already highlighted"
            return out
        if selected != self.GAMEOVER_UNSAFE_ENTRY:
            # Unreadable is not "probably fine": this mode exists to move onto a
            # known entry, so not knowing which one is highlighted is a refusal.
            out["reason"] = "the highlighted entry could not be read"
            return out
        reason = self.gameover_reason() if self.enforce_safety else None
        if reason:
            out["reason"] = reason
            return out
        try:
            self._dispatch(up)
            self._between_presses()
            out["presses"] += 1
            moved = self._poll_safety(
                lambda i: any(e.get("name") == self.GAMEOVER_SAFE_ENTRY
                              for e in (i.get("menuEntries") or [])))
            if not any(e.get("name") == self.GAMEOVER_SAFE_ENTRY
                       for e in (moved.get("menuEntries") or [])):
                out["reason"] = "the cursor did not move onto the save-load entry"
                return out
            self._dispatch(ok)
            self._between_presses()
            for _ in range(max(0, int(max_presses) - 1)):
                out["presses"] += 1
                after = self._poll_safety(lambda i: i.get("scene") != "Scene_Gameover")
                if after.get("scene") != "Scene_Gameover":
                    out["resolved"] = True
                    out["scene"] = after.get("scene")
                    return out
                self._dispatch(ok)
                self._between_presses()
            out["reason"] = "still on the game over screen after %d presses" % out["presses"]
        except (CdpError, UnsafeSceneError) as exc:
            out["reason"] = "refused: %s" % (str(exc)[:120],)
        return out

    def _difficulty_objection(self, info: dict | None) -> str | None:
        """The shared verdict for one safety read; None when a resolve may run."""
        if self.difficulty_option is None:
            return "no difficulty preset configured (--difficulty)"
        if not info:
            return "safety probe unavailable: agent.js not installed"
        if not info.get("awaitingChoice"):
            return "no choice is awaiting an answer"
        if not info.get("guardedChoice"):
            return "the awaiting choice is not the guarded difficulty pick"
        if not info.get("inGameplay"):
            return "scene %r is not Scene_Map gameplay" % (info.get("scene"),)
        if self.drive == "realtime" and not info.get("tickerRunning"):
            return "the engine's ticker is not running"
        return None

    def difficulty_reason(self) -> str | None:
        """None when the preset difficulty may be answered on the agent's behalf.

        Narrower than every other input path in this class: a *guarded* choice is
        awaiting (agent.js's difficulty denylist), the operator configured a
        preset, the scene is Scene_Map, and the engine is running.  A choice that
        is not guarded is refused here, so this can never become a way to answer
        arbitrary dialogue - that is what `unsafe_reason` is for.
        """
        if self.difficulty_option is None:
            return "no difficulty preset configured (--difficulty)"
        try:
            info = self.safety()
        except CdpError as exc:
            return "safety probe failed (%s)" % (str(exc)[:120],)
        return self._difficulty_objection(info)

    def _choice_index_for_option(self, texts: list[str]) -> int | None:
        """Which option the preset names: a 1-based number, or a substring of the text.

        The texts are the *sanitised* ones agent.js reports, so the difficulty
        option that carries a colour code is matchable like any other.
        """
        want = self.difficulty_option
        if want is None:
            return None
        if isinstance(want, bool) or not isinstance(want, (int, str)):
            return None
        if isinstance(want, int):
            i = want - 1               # options are numbered from 1, everywhere
            return i if 0 <= i < len(texts) else None
        text = str(want)
        if text.isdigit():
            i = int(text) - 1          # and a number on the command line means the same
            return i if 0 <= i < len(texts) else None
        for i, option in enumerate(texts):
            if text and text in option:
                return i
        return None

    def resolve_difficulty(self, *, max_presses: int = 8) -> dict[str, Any]:
        """Answer the difficulty pick with the operator's preset option.

        Returns {"resolved", "presses", "option", "reason"}.  This is the only
        path that answers a dialogue choice, and it is deliberately narrow: the
        choice must be classified as guarded, the target is the option the
        operator named, the presses are bounded, and success is *verified* by
        re-reading whether a choice is still waiting rather than assumed.
        """
        out: dict[str, Any] = {"resolved": False, "presses": 0, "option": None,
                               "reason": None}
        # One read for both the verdict and the option texts: two reads could see
        # two different dialogues, and the second one is what would be answered.
        try:
            info = self.safety()
        except CdpError as exc:
            out["reason"] = "safety probe failed (%s)" % (str(exc)[:120],)
            return out
        reason = self._difficulty_objection(info)
        if reason:
            out["reason"] = reason
            return out
        texts = list((info or {}).get("choiceTexts") or [])
        index = self._choice_index_for_option(texts)
        if index is None:
            out["reason"] = ("the preset %r matches none of the options %r"
                             % (self.difficulty_option, texts))
            return out
        out["option"] = index
        current = info.get("choiceIndex")
        if not isinstance(current, int):
            out["reason"] = "the highlighted option could not be read"
            return out
        try:
            down = mask_from_buttons(["down"])
            up = mask_from_buttons(["up"])
            ok = self._confirm_mask()
        except CdpError as exc:
            out["reason"] = str(exc)
            return out
        delta = index - current
        binding = down if delta > 0 else up
        for _ in range(min(abs(delta), max(0, int(max_presses)))):
            self._check_safe(difficulty=True)     # the gate, in its narrow mode
            self._dispatch(binding)
            self._between_presses()
            out["presses"] += 1
        self._check_safe(difficulty=True)
        self._dispatch(ok)
        self._between_presses()
        out["presses"] += 1
        after = self._poll_safety(lambda i: not i.get("awaitingChoice"))
        if after.get("awaitingChoice"):
            out["reason"] = ("the choice is still waiting after %d presses"
                             % out["presses"])
            return out
        out["resolved"] = True
        return out

    def escape_menu(self, *, max_presses: int = 4) -> dict[str, Any]:
        """Press ONLY cancel, at most `max_presses` times, to back out to gameplay.

        Returns {"escaped", "presses", "scene", "reason"}.  It re-reads the scene
        after every press, so it stops the moment gameplay is reached and never
        fires into Scene_Map.
        """
        out: dict[str, Any] = {"escaped": False, "presses": 0, "scene": None, "reason": None}
        try:
            out["scene"] = (self.safety() or {}).get("scene")
        except CdpError:
            pass
        reason = self.menu_escape_reason()
        if reason:
            out["reason"] = reason
            return out
        try:
            mask = self._cancel_mask()
        except CdpError as exc:
            out["reason"] = str(exc)
            return out
        for _ in range(max(0, int(max_presses))):
            self._check_safe(escape=True)      # the gate, in its narrow mode
            out["presses"] += 1
            self._dispatch(mask)
            self._between_presses()
            info = self._poll_safety(lambda i: i.get("inGameplay"))
            out["scene"] = info.get("scene")
            if info.get("inGameplay"):
                out["escaped"] = True
                return out
        out["reason"] = "still not in gameplay after %d cancel press(es)" % out["presses"]
        return out

    def step_realtime(self, action: int, duration_s: float) -> None:
        """Hold an action for `duration_s` of real time, then release it.

        The engine keeps its own loop, so frames advance without help; all this
        does is keep the buttons down long enough for the game to sample them.
        Releasing immediately would let every frame see an idle keyboard (the
        ordering bug `step_frame` documents), and holding across frames is also
        what variable-height jumps need.
        """
        if self.conn is None:
            raise CdpError("not connected")
        self._check_safe()
        bindings = self._bindings_for(action)
        duration = max(0.0, float(duration_s))
        for binding in bindings:
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "rawKeyDown"))
        if self._recording_steps:
            # Keep receiving for the whole hold instead of sleeping through it:
            # the screencast frames of the interval are the consequence of the
            # press, and they are only in the socket while the button is down.
            # Flush first: frames buffered during the previous step's observation
            # belong to the gap, not to this hold.
            self._current_step = None
            self._drain_frames(timeout=0.0)
            self._current_step = []
            start = time.monotonic()
            while True:
                remaining = duration - (time.monotonic() - start)
                if remaining <= 0:
                    break
                # Short drains on purpose: Chromium waits for the frame's ack
                # before it encodes the next one, and a long drain delays that
                # ack.  Measured on this game: ~12 fps with 50 ms drains, ~59 fps
                # with 4 ms - the ack cadence, not the game (59 fps) or the JPEG
                # size (unchanged from 128 to 448 px wide), is what limits it.
                self._drain_frames(timeout=min(0.004, max(0.001, remaining)))
        else:
            time.sleep(duration)
        for binding in bindings:
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "keyUp"))
        if self._current_step is not None:
            self._step_frames.append(self._current_step)
            self._current_step = None

    # ------------------------------------------------------- held input mode
    #
    # Jump height in this game is controlled by how long the button stays down,
    # measured with the engine's own py: a single 250 ms hold (one control
    # interval) reaches 172 px where the game allows 212 px, and the second jump
    # needs a release at the apex before the next press.  A step that always
    # releases therefore cannot express either, whatever the policy wants.
    #
    # So the buttons may stay down ACROSS steps: choosing the same action again
    # extends the hold, and switching action releases it.  "jump, jump" is then a
    # longer jump and "jump, noop, jump" is exactly the game's double jump - both
    # expressible by the policy instead of hard-coded here.
    #
    # `release()` is deliberately not safety gated even though `press()` is:
    # letting go is the safe direction, and refusing to let go inside a menu
    # would leave a button stuck down forever.

    def press(self, mask: int) -> list[Any]:
        """Put an action mask's buttons down and leave them down.  See `release()`.

        Takes a MASK, like `step()` and `apply_action()` do - the action space's
        index is a different number, and passing one where the other belongs
        dispatches the wrong keys (index 3 is mask 3, which is up+down).
        """
        self._check_safe()
        if self._held_action == int(mask):
            return self._held_bindings
        self.release()
        bindings = self._bindings_for(mask)
        for binding in bindings:
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "rawKeyDown"))
        self._held_action = int(mask)
        self._held_bindings = bindings
        return bindings

    def release(self) -> None:
        """Lift whatever `press()` put down.  Idempotent, and never refused."""
        for binding in getattr(self, "_held_bindings", []):
            try:
                self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "keyUp"))
            except CdpError as exc:      # pragma: no cover - transport failure
                log.warning("keyUp failed while releasing: %s", str(exc)[:120])
        self._held_action = None
        self._held_bindings = []

    @property
    def held_action(self) -> int | None:
        return self._held_action

    def advance(self, frames: int = 1) -> int:
        """Let game time pass with the held buttons still down."""
        return self.step_frame(self.action_space.noop, frames=frames, hold=True)

    def step_holding(self, mask: int, frames: int = 1, *,
                     screenshot: bool = True) -> StepResult:
        """One step that does NOT release: the held variant of `step()`.  Mask in."""
        self.press(mask)
        self.advance(frames)
        obs = self.observe(screenshot=screenshot)
        return StepResult(obs=obs, reward=0.0, done=False, info={})

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
        if not hold:
            # A dispatch that is not a hold ends any hold: this is what keeps
            # `press_ok`, `escape_menu`, `apply_action` and the old one-step API
            # from leaving a movement key down while they press something else.
            self.release()
        count = max(1, int(frames))
        if self.drive == "realtime":
            # The engine advances its own frames; hold the buttons for the
            # action's duration of REAL time so it samples them across several
            # of its frames.
            self.step_realtime(action, self.config.frame_ms * count / 1000.0)
            return count
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

    #: Which action's buttons are down right now, and their bindings, so a later
    #: `release()` knows exactly what to lift.
    _held_action: int | None = None
    _held_bindings: list = []

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
        if info.get("awaitingChoice"):
            # Ordinary dialogue is the agent's to operate: the ok key IS the
            # policy's jump key, and choices are how the story advances, so
            # refusing all of them left the agent unable to play the game it is
            # supposed to learn.  The user's rule is narrower - the difficulty
            # pick must never be made by this layer - and that pick is
            # identifiable by its text (V.GUARDED_CHOICES, captured from the
            # running game rather than assumed).
            if info.get("guardedChoice"):
                return ("a dialogue choice that must not be answered by the agent "
                        "is awaiting an answer: %r" % (info.get("choices"),))
            if not info.get("choices"):
                # Fail closed: a choice we cannot read is a choice we cannot
                # classify, and ok would commit whichever option is highlighted.
                return ("a dialogue choice is awaiting an answer but its options "
                        "could not be read (decision point)")
        if self.drive == "realtime" and not info.get("tickerRunning"):
            # Acting against a frozen game is worse than not acting: the round
            # looks healthy, reports steps, and collects a still picture.  That
            # silently wasted a whole verification round.
            return (
                "the engine's ticker is not running, so a realtime round would "
                "act against a frozen game (resume it, or the drive mode is wrong)"
            )
        if info.get("menuOperable"):
            # A menu the agent may actually operate: the shop, the item/equipment
            # screens, the ESC menu itself.  It used to be cancel-only, which made
            # the whole shop unreachable content - the agent could walk in and
            # nothing else.  What keeps this narrow is the entry under the cursor:
            # ok commits it, so a forbidden one is refused, and a scene whose
            # entries cannot be read is refused too (an unclassified entry may be
            # the one that costs the save).
            if info.get("guardedMenuEntry"):
                return ("a menu entry that must not be committed is highlighted: %r"
                        % (info.get("menuEntries"),))
            if not info.get("menuEntries"):
                return ("an operable menu's highlighted entry could not be read "
                        "(decision point)")
            return None
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
        mask = self._confirm_mask()
        for binding in self._bindings_for(mask):
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "rawKeyDown"))
        # The engine samples Input on the frames it runs, so the key has to be
        # DOWN across a frame and released on a later one.  Dispatching down and
        # up back-to-back lets every frame see an idle keyboard and the confirm
        # edge never registers - the same ordering bug step_frame documents.
        self._hold_one_frame()
        for binding in self._bindings_for(mask):
            self.conn.call("Input.dispatchKeyEvent", self._key_params(binding, "keyUp"))
        # The release gets a frame of its own too, so the next press is a fresh
        # edge rather than a continuation of this one.
        self._hold_one_frame()
        return True

    def _hold_one_frame(self) -> None:
        """Leave a key DOWN across a frame the game actually runs.

        The engine samples input on the frames it runs, so a key has to be down
        across one.  In pump mode a pump advances that frame; in realtime mode -
        the only mode live self-play uses (AGENTS.md rule 9) - the engine runs on
        its own clock, so the wait has to be real time and `_pump_frames` is a
        no-op.  `press_ok` relied on the pump anyway, which dispatched keydown and
        keyup in the same millisecond: a live run logged three ok presses five
        milliseconds apart and `Scene_ItemObtain` never cleared, because no frame
        ever saw the key down.
        """
        if self._pump_installed:
            self._pump_frames(1)
            return
        time.sleep(self.config.frame_ms / 1000.0)

    def _pump_frames(self, count: int) -> int:
        """Advance `count` game frames without pressing anything."""
        if self.conn is None:
            raise CdpError("not connected")
        if self._pump_installed:
            return int(
                self.conn.evaluate("window.__ash.pump.pump(%d)" % count, timeout=30.0) or 0
            )
        return 0

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