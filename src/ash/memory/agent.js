/*
 * __ash: the in-page half of the environment layer.
 *
 * Installed with Page.addScriptToEvaluateOnNewDocument so it is present before
 * any game code runs.  It provides exactly three capabilities:
 *
 *   1. a frame pump  - hijacks requestAnimationFrame so the game loop only runs
 *                      when we ask it to, one update at a time (this is what
 *                      makes single-frame stepping and rollback deterministic);
 *   2. state capture - a cycle-safe encoder/decoder for the RPG Maker globals
 *                      that restores in place, preserving object identity;
 *   3. a seeded RNG  - replaces Math.random with a xorshift32 stream so a
 *                      restored frame replays identically.
 *
 * Everything is deliberately defensive: a probe never throws, it records the
 * error, because Phase 0 has to report what is *not* available as clearly as
 * what is.
 */
(function () {
  "use strict";
  // Always replace the agent: an early return when window.__ash exists would
  // keep a STALE version running after the host updated its JS, which is exactly
  // how a fixed bug appears to be unfixed from the outside.  The old pump is
  // left in place on purpose - it references its own closure only, and the new
  // agent installs a fresh one; the game loop simply calls whichever hook is
  // currently bound to requestAnimationFrame.
  window.__ash = undefined;
  delete window.__ash;
  // Remember the real rAF before any pump replaces it, so a repair can put it
  // back even after an older build's uninstall() clobbered it with undefined.
  if (typeof window.requestAnimationFrame === "function" && !window.__ashNativeRAF) {
    window.__ashNativeRAF = window.requestAnimationFrame;
  }
  var V = {};
  window.__ash = V;

  V.version = "0.0.1";
  V.errors = [];

  function guard(name, fn) {
    try { return fn(); } catch (e) { V.errors.push(name + ": " + String(e)); return null; }
  }
  V.guard = guard;

  /* ------------------------------------------------------------------ paths */
  V.defaultPaths = [
    "$gameSystem", "$gameSwitches", "$gameVariables", "$gameSelfSwitches",
    "$gameMap", "$gamePlayer", "$gameParty", "$gameScreen", "$gameTimer",
    "$gameTemp", "$gameMessage"
  ];

  V.globals = function () {
    var out = {};
    var names = V.defaultPaths.concat(["$dataMap", "$dataMapInfos", "SceneManager", "Graphics", "Audio", "Input", "TouchInput"]);
    for (var i = 0; i < names.length; i++) {
      var n = names[i];
      try {
        var v = window[n];
        out[n] = (typeof v === "undefined") ? "undefined" : (v === null ? "null" : typeof v);
      } catch (e) { out[n] = "error:" + String(e); }
    }
    out["scene"] = V.sceneName();
    out["frameCount"] = V.frameCount();
    return out;
  };

  V.sceneName = function () {
    return guard("sceneName", function () {
      var s = window.SceneManager && SceneManager._scene;
      return s ? (s.constructor && s.constructor.name) || "?" : null;
    });
  };

  V.frameCount = function () {
    return guard("frameCount", function () {
      var g = window.Graphics;
      return g ? g.frameCount : null;
    });
  };

  V.playerState = function () {
    return guard("playerState", function () {
      var p = window.$gamePlayer;
      if (!p) { return null; }
      var m = window.$gameMap;
      return {
        x: p.x, y: p.y, direction: p.direction, realX: p._realX, realY: p._realY,
        moveSpeed: p.moveSpeed, through: p._through, transferring: p._transferring,
        mapId: m ? m._mapId : null,
        screenX: p.screenX ? p.screenX() : null,
        screenY: p.screenY ? p.screenY() : null
      };
    });
  };

  V.item = function () {
    return guard("item", function () {
      var p = window.$gameParty;
      if (!p) { return null; }
      var it = p.itemContainer ? p.itemContainer(1) : null;
      return it && it.contents ? it.contents() : null;
    });
  };

  /* ------------------------------------------------------------------- pump */
  V.pump = {};
  V.pump.installed = false;
  V.pump.pending = 0;
  V.pump.time = 0;
  V.pump.dt = 1000 / 60;
  V.pump.hijacked = false;
  V.pump.ticks = 0;

  V.pump.install = function (dtMs) {
    // Attach the fall-recovery marker tracker as soon as the game globals
    // exist; idempotent, and retried on every install until it succeeds.
    V.installGroundingTracker();
    // Heal an event array left broken by an earlier session BEFORE the agent
    // starts driving frames: while it is broken Game_Map.update throws every
    // frame, so nothing the agent does can be trusted.
    V.repairEventGraph();
    if (V.pump.installed) {
      // Do not trust the flag alone.  The engine may have restarted its own
      // loop behind the pump's back (Graphics.startGameLoop -> _app.start),
      // which starts the very ticker install() stopped.  A second install
      // then early-returned and the game kept running at full speed while the
      // agent pumped frames on top of it.  Re-assert ownership every time.
      if (V.pump.mode === "ticker" && V.pump.ticker) {
        V.pump.ticker.autoStart = false;
        V.pump.ticker.stop();
        V.pump.ticker.maxFPS = 0;
        V.pump.guard();
      }
      return true;
    }
    if (dtMs) { V.pump.dt = dtMs; }
    // MZ 1.6+ drives the loop from the PixiJS ticker (Graphics.app.ticker),
    // which captured the real rAF at construction time - replacing
    // window.requestAnimationFrame afterwards cannot intercept it.  The
    // ticker has to be stopped and driven by hand instead.
    if (window.Graphics && Graphics.app && Graphics.app.ticker) {
      V.pump.ticker = Graphics.app.ticker;
      V.pump.ticker.autoStart = false;
      V.pump.ticker.stop();
      V.pump.ticker.maxFPS = 0;
      V.pump.mode = "ticker";
      V.pump.installed = true;
      V.pump.hijacked = true;
      V.pump.guard();
      return true;
    }
    // MV / older MZ: the loop is a rAF callback chain, so the queue works.
    V.pump.origRAF = window.requestAnimationFrame;
    V.pump.origCAF = window.cancelAnimationFrame;
    V.pump.queue = [];
    window.requestAnimationFrame = function (cb) {
      V.pump.queue.push(cb);
      V.pump.pending = V.pump.queue.length;
      return V.pump.queue.length;
    };
    window.cancelAnimationFrame = function () { /* the queue is drained wholesale */ };
    V.pump.mode = "raf";
    V.pump.installed = true;
    V.pump.hijacked = true;
    return true;
  };

  /* Refuse an engine-side restart while the pump owns the loop.
   *
   * MZ restarts its own loop behind the pump's back: Graphics.startGameLoop()
   * (called by SceneManager.run and SceneManager.resume) does
   * Graphics._app.start(), which starts the very ticker install() stopped.
   * stop() cannot see that, so the ticker runs natively at full speed while
   * the agent also pumps frames - the game races ahead uncontrolled and the
   * character moves with nobody driving it.  Shadowing the instance's own
   * start() turns that restart into a no-op for as long as the pump is
   * installed; uninstall()/resume() remove the shadow and the engine gets its
   * start() back.  The shadow is an own property on the ticker instance, which
   * no snapshot root reaches, so a rollback cannot clobber it. */
  V.pump.guard = function () {
    var t = V.pump.ticker;
    if (!t || t.__ashGuarded) { return false; }
    t.__ashStart = t.start;
    // Remember WHERE start() came from.  PIXI puts it on Ticker.prototype, so
    // the shadow is an own property and deleting it reveals the prototype
    // method again.  A ticker that carries its own start() would instead lose
    // the method entirely on delete, so the origin decides how to undo.
    t.__ashOwnStart = Object.prototype.hasOwnProperty.call(t, "start");
    t.start = function () {
      if (V.pump.installed) { return this; }
      return t.__ashStart.apply(this, arguments);
    };
    t.__ashGuarded = true;
    return true;
  };
  V.pump.unguard = function () {
    var t = V.pump.ticker;
    if (!t || !t.__ashGuarded) { return false; }
    if (t.__ashOwnStart) { t.start = t.__ashStart; } else { delete t.start; }
    t.__ashStart = null;
    t.__ashOwnStart = false;
    t.__ashGuarded = false;
    return true;
  };
  /* Hand the game loop back to the engine.
   *
   * This is deliberately NOT what uninstall() does.  A stopped agent must
   * leave the game PAUSED: the character is standing still wherever the run
   * ended, and this engine's enemies track the player relentlessly, so
   * resuming the loop turns "the agent stopped" into "the character is beaten
   * to death while nobody is controlling it".  Freezing is the safe state.
   * Resuming is therefore an explicit, separate call. */
  V.pump.resume = function () {
    var out = { resumed: false, reason: null };
    try {
      var t = V.pump.ticker || (window.Graphics && Graphics.app && Graphics.app.ticker);
      if (!t) { out.reason = "no ticker"; return out; }
      // Release ownership BEFORE starting: the guard refuses start() while
      // the pump is installed, so the flag has to drop first.
      V.pump.installed = false;
      V.pump.hijacked = false;
      V.pump.unguard();
      t.autoStart = true;
      // start() early-returns while started is true, and a ticker can be left
      // started-but-without-a-pending-frame (a zombie); clearing the flag is
      // what makes the call actually re-request a frame.
      t.started = false;
      t.start();
      out.resumed = true;
      return out;
    } catch (e) {
      out.reason = String(e);
      V.errors.push("resume: " + String(e));
      return out;
    }
  };

  V.pump.uninstall = function (resume) {
    if (!V.pump.installed) { return false; }
    if (V.pump.mode === "ticker") {
      // Ticker mode never touched requestAnimationFrame, so restoring the
      // "saved" rAF here would overwrite the real one with undefined (it was
      // never captured) and permanently break every future rAF user.  That
      // corruption is fixed by simply not touching rAF in this mode.
      //
      // The ticker stays STOPPED unless resume is explicitly requested: the
      // game is left frozen with the character safe, which is the whole point
      // of stopping.  Pass resume=true only when a human is about to take over.
      if (resume) { V.pump.resume(); return true; }
      // Freeze for real.  Clearing the flags is not enough: the engine may
      // have restarted the loop behind the pump's back, in which case the
      // ticker is running natively and clearing a flag leaves it running -
      // exactly the reported "the agent stopped but the game is racing at full
      // speed with nobody controlling the character" state.  stop() is
      // idempotent, so it is called unconditionally.
      V.pump.ticker.stop();
      V.pump.unguard();
      V.pump.installed = false;
      V.pump.hijacked = false;
      return true;
    }
    window.requestAnimationFrame = V.pump.origRAF;
    window.cancelAnimationFrame = V.pump.origCAF;
    V.pump.queue = [];
    V.pump.installed = false;
    V.pump.hijacked = false;
    return true;
  };

  /* Run one game-loop iteration; returns the number of callbacks invoked. */
  V.pump.pumpOnce = function () {
    // Late ticker upgrade: install() can run before Graphics.app exists (the
    // page's first document falls back to rAF mode), and MZ creates the
    // renderer only when the scene boots.  Without this, mode stays "rAF" and
    // the ticker runs at full speed on top of the hand-driven queue.  The
    // upgrade mirrors install()'s ticker branch.
    if (V.pump.mode !== "ticker" &&
        window.Graphics && Graphics.app && Graphics.app.ticker) {
      V.pump.ticker = Graphics.app.ticker;
      V.pump.ticker.autoStart = false;
      V.pump.ticker.stop();
      V.pump.ticker.maxFPS = 0;
      V.pump.mode = "ticker";
      V.pump.installed = true;
      V.pump.guard();
    }
    V.pump.time += V.pump.dt;
    V.pump.ticks += 1;
    if (V.pump.mode === "ticker") {
      // A start that slipped through before the guard was applied leaves the
      // ticker running; re-stop it so a frame is never both pumped by hand and
      // driven by the engine's own rAF, which doubles the game speed.
      if (V.pump.ticker.started) { V.pump.ticker.stop(); }
      // PIXI Ticker.update(currentTimeMs) runs exactly one frame: the
      // SceneManager update and the render, in engine order.
      V.pump.ticker.update(V.pump.time);
      return 1;
    }
    var batch = V.pump.queue;
    V.pump.queue = [];
    V.pump.pending = 0;
    var t = V.pump.time;
    for (var i = 0; i < batch.length; i++) {
      try { batch[i](t); } catch (e) { V.errors.push("pumpOnce: " + String(e)); }
    }
    return batch.length;
  };

  V.pump.pump = function (n) {
    var ran = 0;
    for (var i = 0; i < n; i++) { ran += V.pump.pumpOnce(); }
    return ran;
  };

  /* ------------------------------------------------------------------- rng */
  V.rng = {};
  V.rng.state = null;
  V.rng.orig = Math.random;

  V.rng.setSeed = function (seed) {
    var s = (seed >>> 0) || 1;
    V.rng.state = s;
    return s;
  };

  V.rng.next = function () {
    if (V.rng.state === null) { return V.rng.orig(); }
    var x = V.rng.state;
    x ^= (x << 13); x >>>= 0;
    x ^= (x >>> 17);
    x ^= (x << 5); x >>>= 0;
    V.rng.state = x >>> 0;
    return (V.rng.state >>> 0) / 4294967296;
  };

  V.rng.enable = function (seed) {
    if (V.rng.enabled) { if (seed !== undefined) { V.rng.setSeed(seed); } return true; }
    V.rng.setSeed(seed === undefined ? 1 : seed);
    Math.random = function () { return V.rng.next(); };
    V.rng.enabled = true;
    return true;
  };

  V.rng.disable = function () {
    if (!V.rng.enabled) { return false; }
    Math.random = V.rng.orig;
    V.rng.enabled = false;
    return true;
  };

  /* --------------------------------------------------------------- encoder */
  var MAX_DEPTH = 14;
  var SKIP_KEYS = { __ash: 1, __proto__: 1 };

  /* Key order that SURVIVES JSON and matches JS property enumeration.
   *
   * The encoder visited keys with Object.keys(value).sort(), a STRING sort,
   * while JSON.stringify/parse and "for (var k in v)" both enumerate
   * integer-like keys FIRST in NUMERIC order.  For keys ["10","2"] the
   * encoder assigned reference ids as 10-then-2 but the serialized text
   * presented them as 2-then-10, so the decoder consumed the ids in the
   * opposite order: every {__r:N} then resolved to the wrong object, or to
   * nothing at all because that id had not been created yet.  Reproduced
   * offline: three aliases of one object came back as SHARED, undefined,
   * undefined.  That is how a restored Game_Event lost its methods and the
   * game died with "eventIsStarting is not a function". 
   *
   * Integer-like keys are therefore sorted numerically and placed first,
   * exactly as the engine enumerates them, so encoding order, text order and
   * decode order all agree.
   */
  function isArrayIndexKey(k) {
    if (k === "") { return false; }
    if (k.length > 10) { return false; }
    for (var i = 0; i < k.length; i++) {
      var c = k.charCodeAt(i);
      if (c < 48 || c > 57) { return false; }
    }
    if (k.length > 1 && k.charCodeAt(0) === 48) { return false; }
    return Number(k) < 4294967295;
  }
  function canonicalKeys(obj) {
    var keys = Object.keys(obj);
    var numeric = [], rest = [];
    for (var i = 0; i < keys.length; i++) {
      if (isArrayIndexKey(keys[i])) { numeric.push(keys[i]); } else { rest.push(keys[i]); }
    }
    numeric.sort(function (a, b) { return Number(a) - Number(b); });
    rest.sort();
    return numeric.concat(rest);
  }

  function Encoder() {
    this.seen = new Map();
  }

  /* Kept for API compatibility; the encoder no longer pre-numbers anything.
   *
   * Pre-numbering was the BUG, not the fix.  It inserted every reachable
   * object into `seen` before encoding, and encode() begins with
   *     if (this.seen.has(value)) return { __r: ... };
   * so every captured object collapsed to a bare reference and the snapshot
   * carried no state at all: measured 271 bytes for a graph that encodes to
   * 227,532 bytes, with $gamePlayer (289 fields) reduced to { __r: 5 }.
   * restore() then had nothing to write, which is why the player never moved
   * back and why rollback verification reported state_match_rate 0.667 on a
   * snapshot that contained no state.
   *
   * Reference stability is instead guaranteed by encoding keys in SORTED
   * order (see encode).  An in-place restore leaves the target's insertion
   * order scrambled, and a sorted traversal makes id assignment independent
   * of that - which is what canonical numbering was originally trying to
   * achieve, but it broke the encoder to get there.
   */
  Encoder.prototype.canonicalNumbering = function (paths) {
    return;
  };


  Encoder.prototype.encode = function (value, depth) {
    if (value === null) { return null; }
    var t = typeof value;
    if (t === "number") { return (isFinite(value) ? value : { "__num": String(value) }); }
    if (t === "string" || t === "boolean") { return value; }
    if (t === "undefined") { return { "__u": 1 }; }
    if (t === "function") { return { "__f": 1 }; }
    if (t === "symbol") { return { "__y": String(value) }; }
    if (t === "bigint") { return { "__g": String(value) }; }
    if (depth > MAX_DEPTH) { return { "__d": 1 }; }
    if (this.seen.has(value)) { return { "__r": this.seen.get(value) }; }
    // Ids are assigned in traversal order, and the traversal is deterministic
    // because object keys are visited SORTED (see the body loop below).  That
    // sorted order is what keeps reference ids stable across a restore round
    // trip: applyInPlace mutates the target in place, which leaves its key
    // insertion order scrambled, but a sorted walk is independent of
    // insertion order.  This replaces the old canonicalNumbering pre-pass,
    // which pre-registered every object and thereby made encode() emit a bare
    // reference for all of them - a snapshot with no state in it.
    var id = this.seen.size;
    this.seen.set(value, id);
    if (Array.isArray(value)) {
      var arr = [];
      for (var i = 0; i < value.length; i++) { arr.push(this.encode(value[i], depth + 1)); }
      return { "__a": arr };
    }
    if (value instanceof Set) {
      var items = [];
      value.forEach(function (v) { items.push(v); });
      var encSet = [];
      for (var si = 0; si < items.length; si++) { encSet.push(this.encode(items[si], depth + 1)); }
      return { "__set": encSet };
    }
    if (value instanceof Map) {
      var pairs = [];
      value.forEach(function (v, k) { pairs.push([k, v]); });
      var encMap = [];
      for (var mi = 0; mi < pairs.length; mi++) {
        encMap.push([this.encode(pairs[mi][0], depth + 1), this.encode(pairs[mi][1], depth + 1)]);
      }
      return { "__map": encMap };
    }
    if (ArrayBuffer.isView(value)) {
      var ta = [];
      for (var ti = 0; ti < value.length; ti++) { ta.push(value[ti]); }
      return { "__ta": ta, "__c": (value.constructor && value.constructor.name) || "TypedArray" };
    }
    var cls = (value.constructor && value.constructor.name) || "Object";
    var body = {};
    // Canonical order: independent of the object's own insertion order (which
    // an in-place restore scrambles) AND identical to the order JSON and
    // property enumeration use, so reference ids cannot shift across the
    // round trip.  See canonicalKeys.
    var keys = canonicalKeys(value);
    for (var ki = 0; ki < keys.length; ki++) {
      var k = keys[ki];
      if (SKIP_KEYS[k]) { continue; }
      // Skip direct self-references: RPG Maker's $game* objects were observed to
      // accumulate keys like "$gameSwitches" pointing back at themselves after a
      // restore (a restore-time artefact, see applyInPlace).  Encoding them would
      // grow the snapshot on every round trip.
      if (value[k] === value) { continue; }
      body[k] = this.encode(value[k], depth + 1);
    }
    return { "__c": cls, "__v": body };
  };

  /* --------------------------------------------------------------- decoder */
  function Decoder() {
    this.refs = [];
    this.next = 0;
  }

  Decoder.prototype.decode = function (node) {
    if (node === null || typeof node !== "object") { return node; }
    if (Array.isArray(node)) { return this.decode({ "__a": node }); }
    if (node.__u !== undefined) { return undefined; }
    if (node.__f !== undefined) { return function () {}; }
    if (node.__y !== undefined) { return node.__y; }
    if (node.__g !== undefined) { return node.__g; }
    if (node.__num !== undefined) { return Number(node.__num); }
    if (node.__d !== undefined) { return null; }
    if (node.__r !== undefined) { return this.refs[node.__r]; }
    if (node.__a !== undefined) {
      var arr = [];
      this.refs[this.next] = arr; this.next += 1;
      for (var i = 0; i < node.__a.length; i++) { arr.push(this.decode(node.__a[i])); }
      return arr;
    }
    if (node.__set !== undefined) {
      var set = new Set();
      this.refs[this.next] = set; this.next += 1;
      for (var si = 0; si < node.__set.length; si++) { set.add(this.decode(node.__set[si])); }
      return set;
    }
    if (node.__map !== undefined) {
      var map = new Map();
      this.refs[this.next] = map; this.next += 1;
      for (var mi = 0; mi < node.__map.length; mi++) {
        map.set(this.decode(node.__map[mi][0]), this.decode(node.__map[mi][1]));
      }
      return map;
    }
    if (node.__ta !== undefined) {
      var typed = node.__ta.slice();
      this.refs[this.next] = typed; this.next += 1;
      return typed;
    }
    // Restore the prototype: node.__c carries the constructor name recorded by
    // the encoder.  Without this, Game_Item and friends come back as bare
    // Object instances, and their methods (gainItem etc.) disappear.  The
    // prototype is attached WITHOUT calling the constructor: game classes like
    // Game_Event require arguments (mapId, eventId) and their constructors have
    // side effects - the field data below fully determines the restored state.
    var proto = null;
    if (node.__c !== undefined && node.__c !== "Object") {
      try {
        proto = Function("return typeof " + node.__c + " !== 'undefined' ? " +
                         node.__c + ".prototype : null")();
      } catch (e) { proto = null; }
    }
    var obj = proto ? Object.create(proto) : {};
    this.refs[this.next] = obj; this.next += 1;
    var v = node.__v || {};
    // Walk in canonical order, not engine enumeration order, so ids are
    // consumed in exactly the order the encoder assigned them.
    var dk = canonicalKeys(v);
    for (var ki = 0; ki < dk.length; ki++) {
      var k = dk[ki];
      obj[k] = this.decode(v[k]);
    }
    return obj;
  };

  /* In-place restore: mutate target so that it matches the encoded node while
   * preserving the identity of target and of every child object it already has.
   * Identity matters because sprites and the scene hold direct references into
   * the $game* graph; replacing those objects wholesale would desynchronise the
   * rendered scene from the simulated state. */
  function applyInPlace(target, node, dec) {
    if (node === null || typeof node !== "object") { return node; }
    if (node.__r !== undefined) { return dec.refs[node.__r]; }
    if (node.__u !== undefined) { return undefined; }
    if (node.__f !== undefined) {
      // A function placeholder must NEVER overwrite a live function.
      // Returning a fresh noop here replaced every own-property function on
      // the restored object, which is how an experiment that captured
      // `Input` destroyed the input system: Input.update/isTriggered are OWN
      // properties (unlike Game_* methods, which live on the prototype), so
      // they were encoded as {__f:1} and then overwritten with empty
      // functions, leaving `Input.update is not a function` on screen.
      // Keeping the existing target preserves behaviour, and the encoder
      // cannot carry a function body anyway.
      return (typeof target === "function") ? target : function () {};
    }
    if (node.__a !== undefined) {
      if (!Array.isArray(target)) { return dec.decode(node); }
      dec.refs[dec.next] = target; dec.next += 1;
      target.length = node.__a.length;
      for (var i = 0; i < node.__a.length; i++) {
        target[i] = applyInPlace(target[i], node.__a[i], dec);
      }
      return target;
    }
    if (node.__c !== undefined && node.__v !== undefined) {
      if (target === null || typeof target !== "object") { return dec.decode(node); }
      // Keep the prototype in sync with the recorded class.  In-place restore
      // preserves object identity (sprites hold references into this graph), but
      // identity without the right prototype loses all methods: Game_Timer came
      // back as a bare Object, which is what broke the round-trip hash.
      var wantProto = null;
      try {
        wantProto = Function("return typeof " + node.__c + " !== 'undefined' ? " +
                             node.__c + ".prototype : null")();
      } catch (e) { wantProto = null; }
      if (wantProto && !wantProto.isPrototypeOf(target)) {
        Object.setPrototypeOf(target, wantProto);
      }
      dec.refs[dec.next] = target; dec.next += 1;
      var v = node.__v;
      // Canonical order: the reference ids in this body were assigned in
      // that order, so they must be consumed in it too.  Engine enumeration
      // order differs for integer-like keys, which is what misaligned refs
      // (see canonicalKeys).
      var ak = canonicalKeys(v);
      for (var ki = 0; ki < ak.length; ki++) {
        var k = ak[ki];
        target[k] = applyInPlace(target[k], v[k], dec);
      }
      // Remove keys that the snapshot recorded as ABSENT, but never a
      // function.  A blanket delete here removed anything the live object
      // had gained since the snapshot, and the engine injects methods at
      // RUNTIME: eventIsStarting and friends come from the encrypted
      // main.bin, not from any .js file, so they are own properties that
      // appear after a snapshot is taken.  Deleting them produced
      //     eventIsStarting is not a function
      // and killed the game mid-restore on map 14.  Input.update died the
      // same way.  A function that exists now is not game state; the
      // snapshot has no authority to remove it.
      var existing = Object.keys(target);
      for (var ei = 0; ei < existing.length; ei++) {
        var ek = existing[ei];
        if (Object.prototype.hasOwnProperty.call(v, ek)) { continue; }
        if (typeof target[ek] === "function") { continue; }
        delete target[ek];
      }
      return target;
    }
    return dec.decode(node);
  }

  /* Rebuild $gameMap._events slots that are not Game_Events.
   *
   * _events is a 1:1 array of Game_Event indexed by event id, and the engine
   * only ever writes Game_Events into it (nya_physics.js setupEvents, plus the
   * hard-coded map 184/303 slots in nya_patch.js).  Measured live on map 6:
   * _events[13] was the PLAYER OBJECT itself (=== $gamePlayer, eventId() -1),
   * so calcHibernate() put it in _updateEvents too and
   *     Game_Map.updateEventSync -> ev.updateEventSync is not a function
   * threw on EVERY frame.  That aborted Game_Map.update, which froze the map
   * interpreter mid-cutscene (measured: eventId 9, idx 23, waitMode "balloon"),
   * so the story never completed and re-triggered forever, and the game grew
   * Nyaruru/error/ by ~16 files per second.
   *
   * A crashed/scene-frozen game is far worse than one re-created event, but a
   * re-created event is NOT free: `new Game_Event(mapId, id)` initialises at
   * the event's $dataMap tile, so the runtime position and any in-flight move
   * are gone.  Measured on map 6: the rebuilt event 13 sat at its map-data tile
   * (px ~3024) while the running cutscene had scripted the three actors to
   * stand at x = 320 / 416 / 512 - so the actor visibly jumped ~2500px away.
   *
   * The corruption only ever DISPLACES the reference; the original event
   * object usually still exists, held by the sprite set (Sprite_Character keeps
   * a direct reference) or by _updateEvents / _dynamicCharacters.  Recovering
   * that object and putting it back is exact, so it is tried first and the
   * rebuild is only the fallback.
   */
  V.repairEventGraph = function () {
    return guard("repairEventGraph", function () {
      var gmap = window.$gameMap;
      if (!gmap || !gmap._events) { return null; }
      var evs = gmap._events;
      var data = (window.$dataMap && $dataMap.events) ? $dataMap.events : null;
      var repaired = [];

      /* Find the real event object for `id` somewhere else in the graph.
       *
       * Search only places that hold live characters: a stale copy taken out of
       * the graph is still a valid event, and identity is what the sprites and
       * the interpreter bind to. */
      var findOrphan = function (id) {
        var pools = [];
        if (gmap._updateEvents) { pools.push(gmap._updateEvents); }
        if (gmap._dynamicCharacters) { pools.push(gmap._dynamicCharacters); }
        try {
          var sc = SceneManager._scene && SceneManager._scene._spriteset;
          if (sc && sc._characterSprites) {
            for (var si = 0; si < sc._characterSprites.length; si++) {
              var sp = sc._characterSprites[si];
              if (sp && sp._character) { pools.push([sp._character]); }
            }
          }
        } catch (e0) { /* the spriteset is optional */ }
        for (var pi = 0; pi < pools.length; pi++) {
          var pool = pools[pi];
          for (var ii = 0; ii < pool.length; ii++) {
            var cand = pool[ii];
            if (!cand || typeof cand.eventId !== "function") { continue; }
            if (cand === evs[id]) { continue; }
            if (cand === window.$gamePlayer || cand === window.$gameLily) { continue; }
            try { if (cand.eventId() === id) { return cand; } } catch (e1) {}
          }
        }
        return null;
      };

      for (var i = 0; i < evs.length; i++) {
        var e = evs[i];
        if (e === null || e === undefined) { continue; }
        // Method presence, not instanceof: the encrypted main.bin defines some
        // classes lazily, so an instanceof test can be false for a good event.
        if (typeof e.updateEventSync === "function") { continue; }
        var was = (e === window.$gamePlayer) ? "player"
          : ((e === window.$gameLily) ? "lily"
          : (((e && e.constructor && e.constructor.name) || typeof e) + ""));
        var orphan = findOrphan(i);
        if (orphan) {
          evs[i] = orphan;
          repaired.push({ id: i, was: was, fixed: "recovered original object" });
          continue;
        }
        repaired.push({ id: i, was: was, fixed: "rebuilt from $dataMap" });
        if (data && data[i] && typeof Game_Event === "function") {
          evs[i] = new Game_Event(gmap._mapId, i);
        } else {
          evs[i] = null;
        }
      }
      if (repaired.length) {
        // calcHibernate() rebuilds _updateEvents from events(); without it the
        // player object would stay in the update list and keep crashing.
        try {
          if (typeof gmap.calcHibernate === "function") { gmap.calcHibernate(); }
        } catch (e2) { V.errors.push("repairEventGraph/calcHibernate: " + String(e2)); }
        if (gmap._updateEvents && gmap._updateEvents.length) {
          var kept = [];
          for (var u = 0; u < gmap._updateEvents.length; u++) {
            var ue = gmap._updateEvents[u];
            if (ue && typeof ue.updateEventSync === "function") { kept.push(ue); }
          }
          if (kept.length !== gmap._updateEvents.length) {
            gmap._updateEvents = kept;
            repaired.push({ id: -1, was: "_updateEvents filtered" });
          }
        }
        V.lastEventGraphRepair = repaired;
      }
      return repaired;
    });
  };

  /* ------------------------------------------------------------ eval paths */
  V.readPath = function (path) {
    return guard("readPath(" + path + ")", function () {
      var parts = path.split(".");
      var cur = window[parts[0]];
      for (var i = 1; i < parts.length; i++) {
        if (cur === null || cur === undefined) { return null; }
        cur = cur[parts[i]];
      }
      return cur;
    });
  };

  V.snapshot = function (paths) {
    var list = paths || V.defaultPaths;
    // Refs must be STABLE across a restore round trip.  Numbering by traversal
    // order broke: in-place restore visits children in a different order than
    // the fresh encode of the live graph, so "__r: 168" pointed at a different
    // object afterwards and every subsequent hash changed.  The fix is to
    // pre-number the shared objects by walking the graph in a canonical order
    // (per root, in path order) and pass that numbering to the encoder.
    var enc = new Encoder();
    enc.canonicalNumbering(list);
    var gmap = window.$gameMap;
    var gpl = window.$gamePlayer;
    var out = { "__meta": {
      version: V.version,
      frameCount: V.frameCount(),
      scene: V.sceneName(),
      // The map the snapshot belongs to.  $dataMap is deliberately NOT a
      // snapshot root (map data is megabytes), so this is what lets restore()
      // notice that the game has since moved to another map and refuse to
      // splice the two together.  See the guard in V.restore.
      mapId: (gmap && typeof gmap._mapId === "number") ? gmap._mapId : null,
      playerX: (gpl && typeof gpl._realX === "number") ? gpl._realX : null,
      playerY: (gpl && typeof gpl._realY === "number") ? gpl._realY : null,
      rng: V.rng.state,
      rngEnabled: !!V.rng.enabled,
      wallMs: Date.now()
    } };
    var captured = [];
    for (var i = 0; i < list.length; i++) {
      var p = list[i];
      var value = V.readPath(p);
      if (value === null || value === undefined) { continue; }
      out[p] = enc.encode(value, 0);
      captured.push(p);
    }
    out.__meta.captured = captured;
    out.__meta.refs = enc.seen.size;
    return JSON.stringify(out);
  };

  /* Re-enter a map through the engine's own transfer path.
   *
   * Used when a snapshot/restore pair straddles a map change (see the guard
   * in V.restore).  reserveTransfer + SceneManager.goto(Scene_Map) is exactly
   * what a normal in-game transfer does: Scene_Map.create reloads MapXXX.json
   * into $dataMap, Scene_Map.onMapLoaded calls $gamePlayer.performTransfer()
   * (which runs $gameMap.setup on the reloaded data) and then
   * createDisplayObjects() rebuilds the spriteset/tilemap for that map.  It is
   * asynchronous - the map file is fetched off the event loop - so the host
   * pumps frames until $gameMap._mapId and DataManager.isMapLoaded() agree. */
  V.reloadMap = function (mapId, x, y, d) {
    return guard("reloadMap", function () {
      if (!(mapId > 0)) { return false; }
      var px = (typeof x === "number") ? x : 0;
      var py = (typeof y === "number") ? y : 0;
      $gamePlayer.reserveTransfer(mapId, px, py, d || 2, 0);
      SceneManager.goto(Scene_Map);
      return true;
    });
  };

  V.restore = function (json) {
    var root = JSON.parse(json);
    // A snapshot is only meaningful for the map it was taken on.  $dataMap is
    // not captured, so applying an in-place restore after a search rollout
    // walked through a seam puts the OLD $gameMap back (map id, events,
    // terrain) while $dataMap still holds the NEW map's data.  The three
    // pieces - map id, map data, renderer - are then torn apart: the tilemap
    // builds the wrong layers (background goes black so the map looks empty)
    // and the restored terrain no longer matches the ground (the player falls
    // out of the map).  Report the mismatch instead of corrupting; the host
    // reloads the snapshot's map through the engine's own transfer path (see
    // V.reloadMap) and retries.  Measured live: on map 4 a rollout crossed a
    // seam, a later restore left _mapId=4 with a 28x12 "Map005" $dataMap, and
    // the frame went from 64220 colours to 6693 with the player ungrounded.
    var snapMap = (root.__meta && typeof root.__meta.mapId === "number")
      ? root.__meta.mapId : null;
    var curMap = (window.$gameMap && typeof $gameMap._mapId === "number")
      ? $gameMap._mapId : null;
    if (snapMap !== null && curMap !== null && curMap !== snapMap) {
      V.lastRestoreMapMismatch = { want: snapMap, have: curMap };
      return -1;
    }
    var dec = new Decoder();
    var captured = (root.__meta && root.__meta.captured) || [];
    for (var i = 0; i < captured.length; i++) {
      var p = captured[i];
      var parts = p.split(".");
      // Walk to the PARENT of the final segment, starting at window.  The
      // old form started at window[parts[0]] and then took the last segment
      // again, so for a single-segment path like "$gamePlayer" it computed
      //     player["$gamePlayer"] = applyInPlace(player["$gamePlayer"], ...)
      // - assigning to a property that does not exist, while the real player
      // was never touched.  Every captured root is a single-segment path, so
      // NO root was ever restored.  The old all-reference snapshots hid it:
      // applyInPlace returned undefined and the write was harmless.
      var cur = window;
      for (var j = 0; j < parts.length - 1; j++) {
        if (cur === null || cur === undefined) { break; }
        cur = cur[parts[j]];
      }
      if (cur === null || cur === undefined) { continue; }
      var last = parts[parts.length - 1];
      cur[last] = applyInPlace(cur[last], root[p], dec);
    }
    // Cleanup: a restore once left SELF-REFERENCING chains on the captured
    // objects ($gameSystem.$gameSystem. ... = $gameSystem).  With the class-aware
    // applyInPlace above the injection no longer happens on fresh graphs, but a
    // graph polluted by an older agent keeps growing if the cycle is walked, so
    // direct self-references are still removed.  Only key === self is deleted:
    // cross-root references between $game* objects are legal game state and stay.
    var scrubbed = 0;
    for (var ci = 0; ci < captured.length; ci++) {
      var obj = V.readPath(captured[ci]);
      if (obj === null || typeof obj !== "object") { continue; }
      var okeys = Object.keys(obj);
      for (var ok = 0; ok < okeys.length; ok++) {
        if (obj[okeys[ok]] === obj) { delete obj[okeys[ok]]; scrubbed += 1; }
      }
    }
    if (scrubbed) { V.lastSelfRefCleanup = scrubbed; }
    if (root.__meta) {
      if (V.rng.enabled && typeof root.__meta.rng === "number") { V.rng.state = root.__meta.rng >>> 0; }
      var g = window.Graphics;
      if (g && typeof root.__meta.frameCount === "number") { g.frameCount = root.__meta.frameCount; }
    }
    // An in-place restore writes whole subtrees back; if the event array comes
    // back holding something that is not a Game_Event the game dies on the next
    // frame (see repairEventGraph).  Check on every restore, not once.
    V.repairEventGraph();
    return captured.length;
  };

  /* Expose internals for debugging from the host side.  Not used by the game. */
  V.Encoder = Encoder;
  V.Decoder = Decoder;
  V.applyInPlace = applyInPlace;

  /* Repair a ticker left started-without-a-pending-frame by an older build
   * whose uninstall() did not restart it.  requestAnimationFrame is restored
   * to the native implementation first if a previous uninstall clobbered it. */
  V.pump.revive = function () {
    var out = { restoredRAF: false, restartedTicker: false, reasserted: false };
    if (typeof window.requestAnimationFrame !== "function") {
      var native = (window.__ashNativeRAF || null);
      if (native) { window.requestAnimationFrame = native; out.restoredRAF = true; }
    }
    try {
      var t = window.Graphics && Graphics.app && Graphics.app.ticker;
      // While the pump owns the loop the ticker has no pending frame BY
      // DESIGN, so the zombie check below would "repair" the very freeze the
      // agent is deliberately holding: resume() clears the installed flag and
      // hands the loop back, the exact opposite of the paused state a stopped
      // agent must leave behind.  Re-assert ownership instead.
      if (V.pump.installed) {
        out.reasserted = !!V.pump.install();
        return out;
      }
      if (t && (t._requestId === null || t._requestId === undefined)) {
        out.restartedTicker = !!V.pump.resume().resumed;
      }
    } catch (e) { V.errors.push("revive: " + String(e)); }
    return out;
  };

  /* Keep the engine's fall-recovery marker fresh.
   *
   * Game_Player.updateFallDown() reverts to _terrainGroundingX/Y, but the
   * engine only refreshes that marker while the player stands on a
   * Game_Terrain.  Standing on an EVENT - the bed the opening story leaves you
   * on - or falling before ever touching terrain leaves it stale.  Measured:
   * after the walker fell into a hole on map 4 the marker held an air
   * position, so the revert put the player back in mid-air and the fall looped
   * forever ("the character went out of the map"), always staggered and so
   * unable to move out of it.  Refreshing the marker on ANY grounding makes
   * the engine's own recovery land somewhere real again. */
  V.groundingTracker = { installed: false, updates: 0, lastX: 0, lastY: 0 };
  V.installGroundingTracker = function () {
    return guard("installGroundingTracker", function () {
      if (V.groundingTracker.installed) { return true; }
      var proto = window.Game_Player && Game_Player.prototype;
      if (!proto || typeof proto.update !== "function") { return false; }
      var base = proto.update;
      V.groundingTracker.base = base;
      proto.update = function () {
        base.apply(this, arguments);
        if (typeof this.isGrounding === "function" && this.isGrounding()) {
          this._terrainGroundingX = this.px;
          this._terrainGroundingY = this.py;
          V.groundingTracker.updates += 1;
          V.groundingTracker.lastX = this.px;
          V.groundingTracker.lastY = this.py;
        }
      };
      V.groundingTracker.installed = true;
      return true;
    });
  };

  /* Re-arm the fall-recovery marker right now and let the engine take the
   * player back.  Used to break a loop that is already running: the tracker
   * above only helps once the player has grounded again. */
  V.recoverFall = function (x, y) {
    return guard("recoverFall", function () {
      var p = window.$gamePlayer;
      if (!p) { return false; }
      if (typeof x === "number") { p._terrainGroundingX = x; }
      if (typeof y === "number") { p._terrainGroundingY = y; }
      if (typeof p.revertToLastGroundingPos === "function") {
        p.revertToLastGroundingPos();
      }
      return true;
    });
  };

  /* --------------------------------------------------------------- probing */
  V.probe = function () {
    var report = {
      version: V.version,
      hasGameGlobals: false,
      frameCount: V.frameCount(),
      scene: V.sceneName(),
      rng: {
        enabled: !!V.rng.enabled,
        state: V.rng.state,
        native: (Math.random === V.rng.orig)
      },
      pump: { installed: V.pump.installed, hijacked: V.pump.hijacked, ticks: V.pump.ticks },
      errors: V.errors.slice(-20)
    };
    try {
      report.hasGameGlobals = !!(window.SceneManager && window.Graphics && window.$gamePlayer);
      report.globals = V.globals();
    } catch (e) {
      report.errors.push("probe: " + String(e));
    }
    return report;
  };

  /* A snapshot/restore consistency check that can be run entirely in-page:
   * capture, pump n frames, restore, pump the same n frames, compare hashes. */
  V.hashState = function (json) {
    var str = json;
    if (!str) {
      var snap = JSON.parse(V.snapshot());
      // wallMs is Date.now() at capture time - metadata, not game state.  Every
      // snapshot hashes differently while the game is frozen unless it is
      // dropped, which is exactly what made rollback verification meaningless.
      delete snap.__meta.wallMs;
      str = JSON.stringify(snap);
    }
    var h = 2166136261;
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h >>> 0;
  };

  V.selfTest = function (frames) {
    var n = frames || 10;
    var out = {};
    out.pumpInstalled = V.pump.install();
    out.before = V.frameCount();
    V.pump.pump(n);
    out.afterPump = V.frameCount();
    out.framesPerPump = (out.afterPump - out.before) / n;
    var snap = V.snapshot();
    out.snapshotBytes = snap.length;
    V.pump.pump(n);
    var hashA = V.hashState(V.snapshot());
    V.restore(snap);
    V.pump.pump(n);
    var hashB = V.hashState(V.snapshot());
    out.hashA = hashA;
    out.hashB = hashB;
    out.bitExact = (hashA === hashB);
    out.errors = V.errors.slice(-20);
    return out;
  };
})();
