"""Leave the game PAUSED.  Run this before anything that trains.

Why this exists: an unattended game is not a paused one.  The production run
loop pauses between rounds because the ticker only runs inside a round, but every
ad-hoc script that talks to the game has to get this right on its own, and four
of them did not - they called `close(resume=True)`, which is the `ash record`
behaviour where a human is driving.  The cost was measured: a probe left the
character standing in a monster area, hp went 150 -> 0, and the next run found a
GAME OVER screen sitting there.

Idempotent and read-only except for (a) lifting any held button and (b) stopping
the ticker: it never presses a gameplay key.

    uv run python scripts/pause_game.py          # pause it
    uv run python scripts/pause_game.py --status # just report
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.env.cdp_backend import CdpSpeedrunEnv  # noqa: E402


TICKER = ("(function(){try{var t=window.Graphics&&Graphics.app&&Graphics.app.ticker;"
          "return JSON.stringify({started:!!(t&&t.started),"
          " frames:(window.Graphics&&Graphics.frameCount)||null});}catch(e){"
          "return null;}})()")


def _read(env) -> dict:
    import json

    raw = env.conn.evaluate(TICKER)
    return json.loads(raw) if raw else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="report only, never inject")
    args = ap.parse_args()

    env = CdpSpeedrunEnv(drive="realtime", resize=(128, 128))
    try:
        env.connect()
    except Exception as exc:                       # noqa: BLE001 - report, do not raise
        print("游戏未连接（可能已关闭）：%s" % str(exc)[:120])
        return 3

    if args.status:
        # NOT via install(): in realtime mode installing the agent STARTS the
        # engine's ticker, so a "status" that injected would resume the game and
        # then report it as running.  The ticker is readable without the agent.
        import time as _t

        before = _read(env)
        _t.sleep(1.5)
        after = _read(env)
        moving = (before.get("frames") is not None
                  and before.get("frames") != after.get("frames"))
        print("ticker.started=%s | frameCount %s -> %s | %s"
              % (before.get("started"), before.get("frames"), after.get("frames"),
                 "游戏在跑" if moving else "已暂停"))
        env.conn.close()
        env.conn = None
        return 0 if not moving else 1

    # Pausing needs the agent (to lift a held button safely), so it injects -
    # and injecting starts the ticker, which is then stopped and verified, not
    # assumed.  Verification waits: the engine restarts the ticker by itself, so
    # one read right after stop() cannot tell "paused" from "about to resume".
    env.install(seed=0)
    time.sleep(0.3)
    info = env.safety() or {}
    state = env.state()
    print("scene=%s | map=%s | hp=%s" % (info.get("scene"),
                                         (state.get("player") or {}).get("mapId"),
                                         state.get("hp")))
    out = env.pause_game()
    if not out.get("paused"):
        print("!! 暂停失败：%s" % out.get("reason"))
        env.close(resume=False)
        return 1
    print("已暂停并核实保持住了（guarded=%s）。恢复：页面里 __ash.pump.resume()。"
          % out.get("guarded"))
    env.close(resume=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
