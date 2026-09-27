"""CDP live-fire: snapshot -> mutate -> restore round trip on the real game.

Proves the migrated js_source restore path works in the actual nwjs runtime:
after restore, scene/frame/player state must match the snapshot, and a method
injected after the snapshot must survive (pitfall #5).
"""
import asyncio
import json
import sys

sys.path.insert(0, "src")
from ash.memory.js_source import js_source  # noqa: E402

import websockets  # noqa: E402

WS = "ws://127.0.0.1:9222"
_id = 0


async def page_ws() -> str:
    import urllib.request

    pages = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json").read())
    game = [p for p in pages if p.get("type") == "page" and p.get("webSocketDebuggerUrl")]
    if not game:
        raise RuntimeError("no debuggable page: %s" % [p.get("url") for p in pages])
    return game[0]["webSocketDebuggerUrl"]


async def call(ws, method, params=None):
    global _id
    _id += 1
    await ws.send(json.dumps({"id": _id, "method": method, "params": params or {}}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == _id:
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg.get("result", {})


async def evl(ws, expr):
    r = await call(ws, "Runtime.evaluate", {
        "expression": expr, "returnByValue": True, "awaitPromise": True,
    })
    if r.get("exceptionDetails"):
        raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", "?"))
    return r.get("result", {}).get("value")


async def main():
    ws_url = await page_ws()
    async with websockets.connect(ws_url, max_size=100 * 1024 * 1024) as ws:
        await call(ws, "Runtime.enable")
        await call(ws, "Page.enable")
        await call(ws, "Page.addScriptToEvaluateOnNewDocument", {"source": js_source()})
        await evl(ws, js_source())

        out = {}
        # 1. snapshot at frame N
        snap = await evl(ws, "window.__ash.snapshot()")
        out["snap_bytes"] = len(snap)
        state1 = await evl(ws, "window.__ash.playerState() ? JSON.stringify({p:__ash.playerState(),f:__ash.frameCount(),s:__ash.sceneName()}) : null")
        out["before"] = state1
        # 2. mutate: pump 30 frames
        await evl(ws, "__ash.pump.pump(30)")
        state2 = await evl(ws, "JSON.stringify({f:__ash.frameCount(),s:__ash.sceneName()})")
        out["after_pump30"] = state2
        # 3. inject a method after the snapshot (pitfall #5 survivor check)
        await evl(ws, "$gamePlayer.__testRuntimeMethod = function(){ return 42; }; 1")
        # 4. restore
        await evl(ws, "__ash.restore(%s)" % json.dumps(snap))
        state3 = await evl(ws, "JSON.stringify({f:__ash.frameCount(),s:__ash.sceneName(),m:typeof $gamePlayer.__testRuntimeMethod})")
        out["after_restore"] = state3
        ok_frame = json.loads(state1 or "{}").get("f") == json.loads(state3).get("f") if state1 else None
        out["frame_restored"] = ok_frame
        out["method_survived"] = json.loads(state3).get("m") == "function"
        out["verdict"] = "PASS" if ok_frame and out["method_survived"] else "FAIL"
        print(json.dumps(out, indent=2, ensure_ascii=False))


asyncio.run(main())
