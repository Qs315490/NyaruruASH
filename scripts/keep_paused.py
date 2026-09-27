"""Hold the game paused for as long as this process runs.

Why a keeper instead of a single call: pausing does not survive our client
disconnecting.  Measured (2026-09-26):

    same connection:  paused for 20 s, frames frozen at 305568, guarded=true
    after reconnect:  started=true, guarded=false, frames running

The shadow that `__ash.pauseGame()` puts on the ticker's `start()` lives on the
ticker instance, and the game rebuilds that instance; a fresh one has no shadow,
so the engine's own `Graphics._app.start()` restores the loop.  A stopped agent
therefore leaves a character standing in monster areas being killed, which is
exactly what "close(resume=False) leaves the game paused" was supposed to
prevent.

So: run this while training or thinking, and stop it when you want to collect.

    nohup .venv/bin/python -u scripts/keep_paused.py > runs/keep-paused.log 2>&1 &

It never presses a gameplay key; it only re-applies the pause and reports the
frame count, so a frozen game is visible in the log rather than assumed.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.env.cdp_backend import CdpSpeedrunEnv  # noqa: E402

STOP = False


def _stop(*_args: object) -> None:
    global STOP
    STOP = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=2.0, help="seconds between checks")
    ap.add_argument("--minutes", type=float, default=0.0, help="0 = run until killed")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    env = CdpSpeedrunEnv(drive="realtime", resize=(128, 128))
    env.connect()
    env.install(seed=0)
    time.sleep(0.3)
    deadline = None if args.minutes <= 0 else time.time() + args.minutes * 60.0
    frames_seen: int | None = None
    print("holding the game paused (Ctrl-C to stop)", flush=True)

    while not STOP:
        if deadline is not None and time.time() >= deadline:
            break
        try:
            if not getattr(env, "conn", None):
                env.connect()
            # Re-inject if the page lost the agent (a rebuild replaces
            # window.__ash, and with it V.paused).
            alive = env.conn.evaluate("!!(window.__ash && __ash.pauseGame)")
            if not alive:
                env.install(seed=0)
                time.sleep(0.2)
            state = env.ticker_state()
            if state.get("started"):
                out = env.pause_game()
                print("[%s] re-paused: %s" % (time.strftime("%H:%M:%S"), out), flush=True)
            frames = state.get("frames")
            if frames is not None:
                if frames_seen is None:
                    print("[%s] frameCount=%s (baseline)" % (time.strftime("%H:%M:%S"), frames),
                          flush=True)
                elif frames == frames_seen:
                    print("[%s] frameCount=%s frozen" % (time.strftime("%H:%M:%S"), frames),
                          flush=True)
                else:
                    print("[%s] !! frameCount advanced %s -> %s" %
                          (time.strftime("%H:%M:%S"), frames_seen, frames), flush=True)
                frames_seen = frames
        except Exception as exc:               # noqa: BLE001 - keep holding
            print("[%s] lost the page (%s); retrying" % (time.strftime("%H:%M:%S"),
                                                         str(exc)[:100]), flush=True)
            time.sleep(1.0)
            try:
                env.connect()
            except Exception:                  # noqa: BLE001
                pass
        time.sleep(max(0.2, args.every))

    print("stopping the keeper; the game is left paused", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
