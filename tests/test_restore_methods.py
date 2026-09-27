"""Does restore DELETE own properties that appeared after the snapshot?

The live game died with "eventIsStarting is not a function" after a restore.
That symbol is NOT in any game .js file - it comes from the encrypted
main.bin - so it must be attached to objects at runtime.  applyInPlace ends
with a loop that deletes every own key absent from the encoded body:

    var existing = Object.keys(target);
    for (...) if (!hasOwnProperty.call(v, ek)) delete target[ek];

If the live object gained an own method AFTER the snapshot was taken, that
loop removes it - and a removed method is exactly "is not a function".
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ash.memory.js_source import js_source

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required"
)

HARNESS = r"""
const fs = require("fs");
globalThis.window = globalThis;
class Ev { constructor(id){ this.id = id; this.px = 10; this.py = 20; } }
// A real Game_Event always carries updateEventSync (nya_event_trigger.js puts
// it on the prototype), and V.repairEventGraph() keys on exactly that: an
// object in $gameMap._events without it is what made
// Game_Map.updateEventSync throw every frame on map 6.  A stand-in event must
// have it too, or the fixture reads as corruption.
Ev.prototype.updateEventSync = function () {};
class Game_Map { constructor(){ this._mapId = 14; this._events = [ new Ev(1) ]; } }
globalThis.Ev = Ev;
globalThis.Game_Map = Game_Map;
globalThis.$gameMap = new Game_Map();
globalThis.$gamePlayer = { px: 1, py: 2 };
globalThis.$gameSystem = { _f: 0 };
globalThis.$gameSwitches = { _data: [] };
globalThis.$gameVariables = { _data: [] };
globalThis.$gameSelfSwitches = { _data: {} };
globalThis.$gameParty = { _i: {} };
globalThis.$gameScreen = { _b: 255 };
globalThis.$gameTimer = { _f: 0 };
globalThis.$gameTemp = { _d: 0 };
globalThis.$gameMessage = { _t: [] };
globalThis.Graphics = { frameCount: 1, app: { ticker: {
  started: true, autoStart: true, maxFPS: 0, stop(){}, start(){}, update(){} } } };
globalThis.SceneManager = { _scene: { constructor: { name: "Scene_Map" } } };
globalThis.Input = { _s: {}, update(){} };

eval(fs.readFileSync(process.argv[2], "utf8"));
const V = globalThis.__ash;
const out = {};

// Snapshot BEFORE the runtime-injected method exists.
const snap = V.snapshot();
out.methodBeforeSnapshot = typeof $gameMap._events[0].eventIsStarting;

// main.bin attaches it at runtime (simulated as an own property).
$gameMap._events[0].eventIsStarting = function () { return true; };
out.methodAfterInject = typeof $gameMap._events[0].eventIsStarting;

// Restore the earlier snapshot.
V.restore(snap);
out.methodAfterRestore = typeof $gameMap._events[0].eventIsStarting;
out.survived = out.methodAfterRestore === "function";

console.log(JSON.stringify(out));
"""


def test_restore_preserves_runtime_injected_methods(tmp_path: Path):
    harness = tmp_path / "h.js"
    harness.write_text(HARNESS, encoding="utf-8")
    js_file = tmp_path / "i.js"
    js_file.write_text(js_source(), encoding="utf-8")
    proc = subprocess.run(["node", str(harness), str(js_file)],
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    print("\n  method before snapshot:", r["methodBeforeSnapshot"])
    print("  after runtime inject :", r["methodAfterInject"])
    print("  after restore        :", r["methodAfterRestore"])
    assert r["methodAfterInject"] == "function"
    assert r["survived"], (
        "restore DELETED a method injected after the snapshot; that is the "
        "'eventIsStarting is not a function' crash"
    )
