"""The hand-driven clock must live inside the real one.

The pump used to start its synthetic clock at 0 while the ticker's own
`lastTime` was ~40000 ms of real uptime.  PIXI's Ticker.update() runs a frame
only when `currentTime > lastTime`; otherwise it zeroes deltaTime and emits
nothing - so the first pumped frame was silently swallowed - and it then does
`this.lastTime = currentTime`, rebasing the timeline onto the synthetic clock.
Handing the loop back left the engine staring at a ~40 s jump.

These tests load the real agent.js in node against a ticker that records exactly
what it is fed and how it computes deltas, so the property is pinned on the
shipped code rather than on a copy of the reasoning.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

AGENT_JS = Path(__file__).resolve().parent.parent / "src" / "ash" / "memory" / "agent.js"

#: A ticker that behaves like PIXI's: a frame only counts when the time moves
#: forward, and lastTime is always overwritten with whatever it was given.
FAKE = """
function makeTicker(realNow) {
  return {
    started: false, autoStart: true, maxFPS: 0, lastTime: realNow,
    deltas: [], times: [],
    start: function () { this.started = true; return this; },
    stop: function () { this.started = false; return this; },
    update: function (currentTime) {
      this.times.push(currentTime);
      var elapsed = 0;
      if (currentTime > this.lastTime) { elapsed = currentTime - this.lastTime; }
      this.deltas.push(elapsed);
      this.lastTime = currentTime;
    }
  };
}
"""


def _run(body: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    script = (
        "global.window = global;"
        "global.document = {addEventListener: function () {}};"
        "global.navigator = {};"
        "var REAL_NOW = 40000;"                       # page uptime when we take over
        "global.performance = {now: function () { return REAL_NOW; }};"
        + FAKE +
        "global.ticker = makeTicker(REAL_NOW);"
        "global.Graphics = {app: {ticker: global.ticker}};"
        "var SRC = require('fs').readFileSync(%r, 'utf8');"
        "eval(SRC);"
        "var out = (function () { %s })();"
        "console.log(JSON.stringify(out));" % (str(AGENT_JS), body)
    )
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, "node failed:\n%s" % proc.stderr[-900:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_first_pumped_frame_is_not_swallowed():
    """A zero/negative first delta means the game ran no frame at all."""
    out = _run(
        """
        var t = global.ticker;
        __ash.pump.install();
        var lastTimeAfterInstall = t.lastTime;
        __ash.pump.pump(1);
        return {installLastTime: lastTimeAfterInstall, realNow: REAL_NOW,
                firstTime: t.times[0], firstDelta: t.deltas[0], dt: __ash.pump.dt};
        """
    )
    assert out["installLastTime"] == out["realNow"], (
        "install must line the ticker's lastTime up with the real clock"
    )
    assert out["firstDelta"] > 0, "the first pumped frame was swallowed (delta <= 0)"
    assert abs(out["firstDelta"] - out["dt"]) < 1e-6, out


def test_pumped_clock_stays_inside_the_real_timeline():
    """Every pumped time must be within a frame or two of real uptime."""
    out = _run(
        """
        var t = global.ticker;
        __ash.pump.install();
        __ash.pump.pump(120);
        var maxTime = Math.max.apply(null, t.times);
        var n = t.deltas.filter(function (d) { return d > 0; }).length;
        return {maxTime: maxTime, firstTime: t.times[0], count: t.times.length,
                positive: n, realNow: REAL_NOW, dt: __ash.pump.dt};
        """
    )
    assert out["positive"] == out["count"], "no pumped frame may be dropped"
    # 120 frames of 1/60 s is 2 s; the pumped clock must be ~2 s past real uptime,
    # not ~40 s behind it.
    assert out["firstTime"] >= out["realNow"], out
    assert out["maxTime"] - out["realNow"] < out["dt"] * (out["count"] + 2), out


def test_resume_re_anchors_so_the_engine_sees_no_jump():
    """Handing the loop back must not leave the timeline 40 s in the past."""
    out = _run(
        """
        var t = global.ticker;
        __ash.pump.install();
        __ash.pump.pump(30);
        __ash.pump.install();          // re-assert ownership, as a second round does
        __ash.pump.pump(1);
        var beforeResume = t.lastTime;
        __ash.pump.resume();
        var afterResume = t.lastTime;
        // The engine's own next rAF, one frame later: what delta does it see?
        t.update(afterResume + 17);
        var engineDelta = t.deltas[t.deltas.length - 1];
        return {beforeResume: beforeResume, afterResume: afterResume,
                engineDelta: engineDelta, realNow: REAL_NOW, started: t.started,
                dt: __ash.pump.dt};
        """
    )
    assert out["started"] is True, "resume must start the engine's loop"
    # lastTime stays at the last pumped instant, which is *inside* the real
    # timeline (real uptime plus the pumped span) - not 0 and not 40 s behind.
    assert out["afterResume"] >= out["realNow"], out
    assert out["afterResume"] - out["realNow"] < out["dt"] * 32, out
    # And what the engine sees next must be an ordinary frame, not a huge jump.
    assert 0 < out["engineDelta"] <= out["dt"] * 2, out
