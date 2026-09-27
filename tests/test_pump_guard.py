"""A guard left behind by an older agent instance must still be removable.

Re-injecting agent.js replaces ``window.__ash`` (and the closure variable
``V``), but the shadow ``start()`` the pump puts on the *ticker instance*
survives - it lives on the ticker object, not on the agent.  The next agent
then held no handle to that ticker, so ``unguard()`` silently did nothing and
``resume()`` reported ``{resumed: true}`` while the ticker stayed stopped and
shadowed: the game frozen for good, with ``close(resume=True)`` and
``resume_game()`` both claiming success.

Measured on the real game before the fix: ``install()`` in pump mode, then a
re-inject, then ``resume()`` -> ``{resumed: true}`` and ``ticker.started``
still ``false``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

AGENT_JS = Path(__file__).resolve().parent.parent / "src" / "ash" / "memory" / "agent.js"


def _run(body: str) -> dict:
    """Load the real agent.js in node against a fake PIXI ticker, then run body."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    script = (
        # A ticker with just the surface the pump uses.
        "function makeTicker() {"
        "  return {started: false, autoStart: true, maxFPS: 0, _requestId: null,"
        "    start: function () { this.started = true; return this; },"
        "    stop: function () { this.started = false; return this; }};"
        "}"
        "global.PIXI = {Ticker: {prototype: {start: makeTicker().start}}};"
        "global.ticker = makeTicker();"
        "global.window = global;"
        "global.document = {addEventListener: function () {}};"
        "global.navigator = {};"
        "global.Graphics = {app: {ticker: global.ticker}};"
        "var SRC = require('fs').readFileSync(%r, 'utf8');"
        "eval(SRC);"
        "var out = (function () { %s })();"
        "console.log(JSON.stringify(out));" % (str(AGENT_JS), body)
    )
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, "node failed:\n%s" % proc.stderr[-800:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_resume_recovers_a_guard_left_by_an_older_agent():
    out = _run(
        """
        var t = global.ticker;
        __ash.pump.install();               // pump owns the loop
        var guarded = !!t.__ashGuarded;
        var stoppedByPump = t.started === false;
        delete window.__ash;                // the next round reinjects agent.js
        eval(SRC);                          // fresh V, no handle to the ticker
        var freshKnowsTicker = !!__ash.pump.ticker;
        var resumed = __ash.pump.resume();
        return {guarded: guarded, stoppedByPump: stoppedByPump,
                freshKnowsTicker: freshKnowsTicker,
                resumed: resumed, started: t.started, stillGuarded: !!t.__ashGuarded};
        """
    )
    assert out["guarded"], "the pump must have shadowed start()"
    assert out["stoppedByPump"], "the pump must have stopped the ticker"
    assert not out["freshKnowsTicker"], "the fresh agent has no ticker handle"
    assert out["resumed"]["resumed"] is True, out
    assert out["started"] is True, "resume() must actually start the ticker"
    assert not out["stillGuarded"], "the stale guard must be removed"


def test_resume_reports_failure_instead_of_claiming_success():
    """An unstartable ticker must not be reported as resumed."""
    out = _run(
        """
        var t = global.ticker;
        delete t.start;                     // nothing left that can start it
        return {resumed: __ash.pump.resume(), started: t.started};
        """
    )
    assert out["resumed"]["resumed"] is False, out
    assert out["resumed"]["reason"], "a refusal must say why"


def test_a_live_agent_still_holds_the_loop_while_the_pump_is_installed():
    """The guard's whole purpose: refuse start() while the pump owns the loop."""
    out = _run(
        """
        var t = global.ticker;
        __ash.pump.install();
        var started = t.start();            // must be refused
        return {startedFlag: t.started, installed: __ash.pump.installed};
        """
    )
    assert out["installed"] is True
    assert out["startedFlag"] is False, "start() must stay a no-op under the pump"
