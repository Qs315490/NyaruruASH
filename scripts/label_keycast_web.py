"""Browser-based keycast labeler: the Tk version could not render text at all.

Measured on this machine, which is why this exists:

    Tk 9.0.4, windowingsystem=x11
    tkinter.font.families()  ->  ('fixed',)
    font "Noto Sans CJK JP" resolves to "fixed"
    font.measure("中文")     ->  0        (zero width: nothing is drawn)
    xlsfonts                 ->  0 core fonts
    system python3's Tk      ->  ImportError: libtk8.6.so

So every label in the Tk UI was invisible, ASCII included - reported first as
"some buttons have no text" and then as "Chinese does not show".  That is an
environment problem, not a translation one, and no amount of font configuration
fixes a Tk that cannot see any font.  A browser renders text with its own font
stack (this box has Noto Sans CJK), so the UI moves there.

The page also takes over video decoding: playback, seeking and speed are the
browser's, which is smoother than streaming decoded frames from Python, and the
magnified panel is just the same video drawn into a small canvas at a larger size.

    uv run python scripts/label_keycast_web.py --video data/video-src/BV19s4y1y7un.mp4

Then open the printed URL (it also tries to launch a browser).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ash.data.video_pack as _vp  # noqa: E402
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: cells measured on BV19s4y1y7un; per video they must be re-measured.
#: Serialises file writes.  ThreadingHTTPServer handles each request in its own
#: thread, so two overlapping posts (a drag and a click, say) could interleave
#: write_text's truncate-then-write and leave two JSON documents in one file - which
#: is exactly what happened: the next start could not parse the cell list, fell back
#: to the built-in one, and directions died with it.
_WRITE_LOCK = threading.Lock()


def atomic_write(path: Path, obj: dict) -> None:
    """Write JSON so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _WRITE_LOCK:
        tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
        os.replace(tmp, path)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:                                 # noqa: BLE001
        return {}


DEFAULT_CELLS = {
    "L": (37, 69), "R": (78, 69), "U": (57, 49), "D": (58, 89),
    "c1": (134, 63), "c2": (129, 102), "c3": (167, 47),
    "c4": (203, 47), "c5": (213, 13), "c6": (245, 13),
}
PANEL = (250, 582)
PANEL_W, PANEL_H = 300, 138

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
 :root { color-scheme: dark; }
 body { background:#14161a; color:#e8e8e8; font: 14px/1.5 "Noto Sans CJK SC",
        "Noto Sans CJK JP", system-ui, sans-serif; margin:0; padding:10px; }
 .row { display:flex; gap:14px; align-items:flex-start; }
 .col { display:flex; flex-direction:column; gap:8px; }
 canvas { background:#000; border:1px solid #333; border-radius:4px; }
 button, select { background:#23262d; color:#e8e8e8; border:1px solid #3a3f48;
        border-radius:4px; padding:5px 10px; font:inherit; cursor:pointer; }
 button:hover { background:#2c313a; }
 input[type=range] { width: 100%; }
 table { border-collapse: collapse; }
 td, th { padding:3px 6px; text-align:left; }
 tr.sel { background:#2a2f38; }
 .down { color:#6cf06c; font-weight:700; }
 .unknown { color:#888; }
 .pos { color:#ffd479; font-variant-numeric: tabular-nums; }
 #panel { image-rendering: pixelated; }
 #status { color:#9ad; min-height:1.5em; }
 label { color:#9aa4b2; }
</style></head><body>
<h3 id="h">__H__</h3>
<div class="row">
  <div class="col">
    <canvas id="video" width="1280" height="720"></canvas>
    <div>
      <button id="b-start"></button><button id="b-back"></button>
      <button id="b-prev"></button><button id="b-play"></button>
      <button id="b-next"></button><button id="b-fwd"></button>
      <label id="l-speed"></label>
      <select id="speed"><option>0.25</option><option>0.5</option>
        <option selected>1</option><option>2</option><option>4</option></select>
      <label id="l-zoom"></label>
      <select id="zoom"><option value="0.25">25%</option><option value="0.33">33%</option>
        <option value="0.5">50%</option><option value="0.75">75%</option>
        <option value="1" selected>100%</option><option value="1.5">150%</option></select>
      <label id="l-lang"></label><select id="lang">
        <option value="zh">中文</option><option value="en">English</option></select>
    </div>
    <div id="hint" class="pos"></div>
    <input id="seek" type="range" min="0" max="1000" value="0" step="1">
    <div class="pos" id="pos"></div>
    <div id="status"></div>
  </div>
  <div class="col">
    <div id="l-panel"></div>
    <div class="pos" id="panelinfo"></div>
    <div class="pos" id="handle"></div>
    <canvas id="panel" width="600" height="276"></canvas>
    <div id="l-table"></div>
    <table id="map"></table>
    <div><button id="b-save"></button><button id="b-scan"></button>
      <button id="b-discover"></button></div>
  </div>
</div>
<video id="v" src="/video.mp4" style="display:none" preload="auto"></video>
<canvas id="probe" width="__PW__" height="__PH__" style="display:none"></canvas>
<script>
// Any error in this script leaves the page as an empty HTML skeleton with no clue why -
// which is exactly how "the page only shows its frame" was reported.  Make the failure
// speak instead: the message lands in the status line where the user is already looking.
window.addEventListener('error', ev => {
  const st = document.getElementById('status');
  if(st) st.textContent = "JS error: " + (ev.message || (ev.error && ev.error.message) || ev);
});
let CELLS = __CELLS__;
let PANEL = __PANEL__, PW = __PW__, PH = __PH__;
// Native frame size.  Everything - panel box, cell coordinates, hit testing, clamping -
// lives in this space, because that is where the boxes and cells were measured.  The
// first version hardcoded 1280x720: for a 1920x1080 video the panel (y=815) and its
// keycaps (y~884) were simply outside the canvas, so the box could not be grabbed by its
// bottom edge (it was off-screen) and every marker was drawn in the wrong place.  The
// display size is CSS's job now, not the coordinate system's.
let VW = 1280, VH = 720;
// The panel box is per video and a screenshot estimate can be wrong by enough to
// slice the row above into the crop (that is what broke one video's layout).  The
// box is therefore adjustable by dragging a rectangle on the frame.
let panelMode = false, panelRect = null;
let NAMES = Object.keys(CELLS), NUM = {};
NAMES.forEach((k,i)=>NUM[k]=i+1);
const S = {
 zh:{h:"速通按键标注（浏览器版）",start:"回到开头",back:"后退5秒",prev:"上一帧",
     play:"播放",pause:"暂停",next:"下一帧",fwd:"前进5秒",speed:"速度",lang:"语言",
     zoom:"视频缩放",panel:"按键面板（放大）",table:"细胞 → 物理键",num:"#",cell:"细胞",key:"物理键",
     state:"状态",stat:"统计",down:"按下",up:"·",ignore:"（忽略）",save:"保存映射",
     saved:"已保存",need:"未判定",presses:"按下 %d 次",title:"速通按键标注",
     prevPress:"上一处按下",nextPress:"下一处按下",scan:"扫描全片",
     scanning:"扫描中 %d%%",scanDone:"扫描完成（每格按下次数已更新）",
     noNext:"%s 之后没有按下",noPrev:"%s 之前没有按下",jumpTo:"已跳到 %.2fs",
     hint:"快捷键：空格 播放/暂停 · ←/→ ±5秒 · Shift+←/→ 单帧 · , . 单帧 · Home 回到开头 · Shift+拖拽视频＝重画面板框",
     addCell:"新增细胞 %s（点击放大面板添加；拖动已有标记可改位置）",del:"删除",deleted:"已删除 %s",
     moved:"%s 已移到 %s",
     scanStop:"停止扫描",scanStopped:"扫描已停止（已扫到的区间保留）",
     panelSet:"框选面板位置",panelHint:"在视频上拖一个矩形框住按键面板（Esc 取消）",
     panelDone:"面板已设为 (%d,%d) 尺寸 %dx%d；细胞已随面板一起平移",
     panelNow:"当前面板 (%d,%d) %dx%d",
     dirMoved:"方向九宫格已整体移动（9 个球位一起）",
     discover:"扫描缺失按键",discovering:"扫描中 %d%%（从当前帧起，每秒一帧）",
     discoverDone:"扫描完成：发现 %d 个候选位置，新增 %d 个细胞",
     discoverStopped:"已停止：已扫到的部分已处理，新增 %d 个细胞",
     discoverNone:"没有发现缺失的键帽（已有细胞覆盖了所有亮过的位置）"},
 en:{h:"keycast labeler (browser)",start:"start",back:"-5s",prev:"prev frame",
     play:"play",pause:"pause",next:"next frame",fwd:"+5s",speed:"speed",lang:"lang",
     zoom:"zoom",panel:"keycast panel (magnified)",table:"cell → physical key",num:"#",cell:"cell",
     key:"key",state:"state",stat:"stats",down:"DOWN",up:"·",ignore:"(ignore)",
     save:"save mapping",saved:"saved",need:"n/a",presses:"%d presses",title:"keycast labeler",
     prevPress:"prev press",nextPress:"next press",scan:"scan whole video",
     scanning:"scanning %d%%",scanDone:"scan done (press counts updated)",
     panelSet:"set panel box",panelHint:"drag a rectangle over the keycast panel (Esc cancels)",
     panelDone:"panel (%d,%d) %dx%d; cells shifted with it",
     panelNow:"panel now (%d,%d) %dx%d",
     dirMoved:"direction grid moved as a whole (all nine ball positions)",
     discover:"find missing keys",discovering:"scanning %d%% from here (one frame per second)",
     discoverDone:"done: %d candidate positions, %d cells added",
     discoverStopped:"stopped: what was scanned is kept, %d cells added",
     discoverNone:"no missing keycaps found (the existing cells cover everything lit)",
     noNext:"%s has no press after this",noPrev:"%s has no press before this",
     jumpTo:"jumped to %.2fs",
     hint:"keys: space play/pause · ←/→ ±5s · shift+←/→ one frame · , . one frame · home start · shift+drag frame = redraw panel box",
     addCell:"added cell %s (click the panel to add; drag a marker to move it)",
     moved:"%s moved to %s",
     del:"delete",deleted:"deleted %s",scanStop:"stop scan",
     scanStopped:"scan stopped (intervals found so far are kept)"}
};
const DIRPROTO = __DIRPROTO__;   // 8 directions + neutral, measured from the video
const DIRSET = {U:["U"],UL:["U","L"],UR:["U","R"],L:["L"],"-":[],R:["R"],
                DL:["D","L"],DR:["D","R"],D:["D"]};
const KEYS = __KEYS__;
const KEYLABEL = {zh:{up:"上 ↑",down:"下 ↓",left:"左 ←",right:"右 →",jump:"跳跃 Z",
  attack:"攻击 X",dash:"冲刺 C",special:"咸鱼技能 V",interact:"确定/交互",menu:"菜单",
  ult:"大招 A",weapon_switch:"换武器 S",cancel:"取消 X",item:"用物品 F"},en:{}};
let lang = "zh", mapping = __MAPPING__, selected = null, t = {};
const v = document.getElementById('v');
const cv = document.getElementById('video'), cx = cv.getContext('2d');
const pc = document.getElementById('panel'), px = pc.getContext('2d');
const pr = document.getElementById('probe'), pxx = pr.getContext('2d');
const stat = {}, known = {}, events = {}, open = {};
NAMES.forEach(k=>{stat[k]={lo:255,hi:0,down:false,n:0};known[k]=false;
                  events[k]=[];open[k]=null;});
function syncCells(){          // after adding/removing a keycap by hand
  NAMES = Object.keys(CELLS); NUM = {};
  NAMES.forEach((k,i)=>{ NUM[k]=i+1;
    if(!stat[k]) stat[k]={lo:255,hi:0,down:false,n:0};
    if(known[k]===undefined) known[k]=false;
    if(!events[k]) events[k]=[];
    if(open[k]===undefined) open[k]=null; });
}
function postCells(){
  fetch('/api/cells',{method:'POST',
    body:JSON.stringify({cells:CELLS, panel:PANEL, panel_size:[PW,PH],
                         dir_proto:DIRPROTO})});
}
function applyPanel(x, y, w, h){
  // The box and the cells are ONE template: moving it carries the cells along, resizing
  // it scales them.  The first version shifted the cells so that their ABSOLUTE frame
  // positions were preserved - correct for "crop a panel whose keycaps I already know",
  // wrong for "fit this layout onto the panel on screen", which is what the box is for:
  // there the markers must stay on the same spot of the panel graphic.
  const sx = w / PW, sy = h / PH;
  for(const k in CELLS){ CELLS[k] = [Math.round(CELLS[k][0] * sx), Math.round(CELLS[k][1] * sy)]; }
  // dir_proto lives in the same panel-relative space and must follow, or the direction
  // decoder silently reads nine positions that no longer line up with the ball.
  for(const k in DIRPROTO){ DIRPROTO[k] = [DIRPROTO[k][0] * sx, DIRPROTO[k][1] * sy]; }
  PANEL = [x, y]; PW = w; PH = h;
  pr.width = PW; pr.height = PH;
  pc.width = PW * 2; pc.height = PH * 2;   // magnified view follows the box
  _tag = new Uint8Array(PW * PH); _lab = new Int32Array(PW * PH);
  postCells(); buildTable();
  document.getElementById('status').textContent =
    S[lang].panelDone.replace("%d", x).replace("%d", y).replace("%d", w).replace("%d", h);
}
const COLOUR = {L:"#ff5050",R:"#50ff50",U:"#ffff50",D:"#ff8c50"};
const DIR = ["L","R","U","D"];

function keyLabel(k){ if(k==="(ignore)") return t.ignore;
  return (KEYLABEL[lang]||{})[k] || k; }
function keyCanon(label){ if(label===t.ignore) return "(ignore)";
  for(const k of KEYS) if(keyLabel(k)===label) return k;
  return "(ignore)"; }

let _tag = new Uint8Array(PW * PH), _lab = new Int32Array(PW * PH);
let _blobs = [];
function components(){
  // Connected components of the bright pixels, once per frame.  Judging each cell
  // independently from a patch average was the source of the crosstalk: where a
  // neighbour's disc reached into a cell's patch the value rose, and the hysteresis
  // then LATCHED it on.  With blobs, a keycap presses exactly the cells whose centre
  // lies inside its own disc, so a neighbour cannot claim it.
  // Draw THIS frame into the probe canvas first.  Splitting the old measure() into
  // components() dropped this line, so getImageData kept returning the last frame
  // ever drawn - an empty canvas at start - and no key was ever seen.
  pxx.drawImage(v, PANEL[0], PANEL[1], PW, PH, 0, 0, PW, PH);
  const img = pxx.getImageData(0, 0, PW, PH).data;
  const N = PW * PH;
  for(let i = 0; i < N; i++){
    const g = (img[i*4] + img[i*4+1] + img[i*4+2]) / 3;
    _tag[i] = g > 150 ? 1 : 0;
    _lab[i] = 0;
  }
  _blobs = [];
  const stack = [];
  for(let i = 0; i < N; i++){
    if(_tag[i] !== 1 || _lab[i] !== 0) continue;
    const id = _blobs.length + 1;
    let area = 0, sx = 0, sy = 0;
    stack.length = 0; stack.push(i); _lab[i] = id;
    while(stack.length){
      const p = stack.pop(); const x = p % PW, y = (p / PW) | 0;
      area++; sx += x; sy += y;
      if(x > 0    && _tag[p-1]  === 1 && _lab[p-1]  === 0){ _lab[p-1]  = id; stack.push(p-1); }
      if(x < PW-1 && _tag[p+1]  === 1 && _lab[p+1]  === 0){ _lab[p+1]  = id; stack.push(p+1); }
      if(y > 0    && _tag[p-PW] === 1 && _lab[p-PW] === 0){ _lab[p-PW] = id; stack.push(p-PW); }
      if(y < PH-1 && _tag[p+PW] === 1 && _lab[p+PW] === 0){ _lab[p+PW] = id; stack.push(p+PW); }
    }
    if(area >= 100) _blobs.push({id, area, cx: sx/area, cy: sy/area});
  }
  return _blobs;
}

function measure(){
  const all = components();
  // A keycap is 250-1400 px at this scale.  Anything smaller is a speck
  // (antialiasing, noise, bleed) and anything larger is overlay furniture - and
  // `_lab` holds EVERY component, including the specks that `components()` already
  // filtered out of `_blobs`.  Testing membership in `_lab` alone therefore let a
  // one-pixel speck press a key, which is what produced both the crosstalk and the
  // press nobody could explain.
  // Keycap size scales with the panel: the limits were hardcoded for a 720p panel
  // (a keycap is ~700 px there), so on the 1080p video - native pixels, keycap ~1600-2000 -
  // every blob fell outside the ceiling and was filtered out.  That is silent: presses
  // simply never register and the keycap scan finds nothing.
  const K = PW / 300;
  const KEY_MIN = 150 * K * K, KEY_MAX = 1600 * K * K;
  const blobs = all.filter(b => b.area >= KEY_MIN && b.area <= KEY_MAX);
  const byId = new Map(blobs.map(b => [b.id, b]));
  // --- direction: the ball is the big blob in the left third ------------------
  let ball = null;
  for(const b of blobs) if(b.cx < 110 && (!ball || b.area > ball.area)) ball = b;
  let hit = "-";
  if(ball){
    let bd = 1e9;
    for(const k in DIRPROTO){
      const p = DIRPROTO[k], dd = (p[0]-ball.cx)**2 + (p[1]-ball.cy)**2;
      if(dd < bd){ bd = dd; hit = k; }
    }
  }
  // --- buttons: a cell is down when its centre sits in its own bright disc ----
  // If there is no ball (this video's keycast draws the ARROW KEYS as keycaps instead),
  // the direction cells are ordinary keycaps and are detected like every other one.
  const useBall = Object.keys(DIRPROTO).length > 0;
  const want = new Set();
  for(const k of NAMES){
    if(DIR.includes(k) && useBall) continue;
    const x = Math.round(CELLS[k][0]), y = Math.round(CELLS[k][1]);
    let on = false;
    for(let dy = -2; dy <= 2 && !on; dy++) for(let dx = -2; dx <= 2 && !on; dx++){
      const xx = x + dx, yy = y + dy;
      if(xx < 0 || yy < 0 || xx >= PW || yy >= PH) continue;
      if(byId.has(_lab[yy * PW + xx])) on = true;
    }
    if(on) want.add(k);
  }
  for(const k of NAMES){
    const s = stat[k];
    if(DIR.includes(k) && useBall){
      // two samples in a row: the centroid can land between two positions for one frame
      s.pend = DIRSET[hit] && DIRSET[hit].includes(k) ? (s.pend || 0) + 1 : 0;
      if(s.pend >= 2 && !s.down){ s.down = true; s.n++; }
      else if(s.pend === 0 && s.down){ s.down = false; }
      known[k] = true;
    } else {
      s.pend = want.has(k) ? (s.pend || 0) + 1 : 0;
      if(s.pend >= 2 && !s.down){ s.down = true; s.n++; }
      else if(s.pend === 0 && s.down){ s.down = false; }
      known[k] = true;
    }
  }
  // Record the transitions ONCE, after both groups are decided: the jump buttons
  // and the press counters must come from the same intervals.
  for(const k of NAMES){
    const s = stat[k];
    if(s.down && open[k] === null) open[k] = v.currentTime;
    else if(!s.down && open[k] !== null){ events[k].push([open[k], v.currentTime]); open[k] = null; }
  }
  stat.__ball = hit;
}

function draw(){
  cx.drawImage(v, 0, 0, VW, VH);
  if(panelRect){
    cx.strokeStyle = "#ffd479"; cx.lineWidth = 2;
    cx.strokeRect(panelRect.x0, panelRect.y0, panelRect.x1 - panelRect.x0,
                  panelRect.y1 - panelRect.y0);
  }
  // Show the panel box itself.  Without it there is no way to see what the page thinks
  // the panel is - and the page's idea of the box is the thing that decides whether the
  // magnified crop and the markers can be right at all.
  cx.strokeStyle = "#ffd479"; cx.lineWidth = 2; cx.setLineDash([6, 4]);
  cx.strokeRect(PANEL[0], PANEL[1], PW, PH);
  cx.setLineDash([]);
  for(const k of NAMES){
    // CELLS are PANEL-relative.  The panel view starts at the panel origin, so
    // CELLS*2 is right there, but the video canvas starts at the frame origin:
    // without adding PANEL every marker landed in the top-left corner instead of
    // on its keycap.
    const p = [PANEL[0] + CELLS[k][0], PANEL[1] + CELLS[k][1]], on = stat[k].down;
    const col = known[k] ? (COLOUR[k]||"#3cf") : "#888";
    cx.beginPath(); cx.arc(p[0],p[1],12,0,6.284);
    cx.strokeStyle=col; cx.lineWidth=2; cx.stroke();
    if(on){ cx.fillStyle=col; cx.fill(); }
    if(k===selected){ cx.beginPath(); cx.arc(p[0],p[1],16,0,6.284);
      cx.strokeStyle="#fff"; cx.lineWidth=2; cx.stroke(); }
    cx.fillStyle = on ? "#000" : col;
    cx.font = "bold 15px sans-serif"; cx.fillText(String(NUM[k]), p[0]-5, p[1]+5);
  }
  const psx = pc.width / PW, psy = pc.height / PH;
  px.drawImage(v, PANEL[0], PANEL[1], PW, PH, 0, 0, pc.width, pc.height);
  for(const k of NAMES){
    const x = CELLS[k][0] * psx, y = CELLS[k][1] * psy, on = stat[k].down;
    const col = known[k] ? (COLOUR[k]||"#3cf") : "#888";
    px.beginPath(); px.arc(x,y,14,0,6.284); px.strokeStyle=col; px.lineWidth=2; px.stroke();
    if(on){ px.fillStyle=col; px.fill(); }
    if(k===selected){ px.beginPath(); px.arc(x,y,19,0,6.284);
      px.strokeStyle="#fff"; px.lineWidth=2; px.stroke(); }
    px.fillStyle = on ? "#000" : col; px.font="bold 15px sans-serif";
    px.fillText(String(NUM[k]), x-6, y+6);
  }
  // Drawn LAST, on top of the crop: this was drawn before the crop for a moment, which
  // painted it straight over - so the box existed, tracked the grid, responded to drags,
  // and was invisible.
  const db = dirBox();
  if(db){
    px.strokeStyle = "#6cf"; px.lineWidth = 2; px.setLineDash([5, 4]);
    px.strokeRect(db.x0 * psx, db.y0 * psy, (db.x1 - db.x0) * psx, (db.y1 - db.y0) * psy);
    px.setLineDash([]);
  }
}

let drag = null;
let dirDrag = null;
const DIRBOX_PAD = 10;
function dirBox(){
  // The 3x3 grid gets its own box, so it can be STRETCHED onto the ball's positions --
  // moving it alone was not enough when the panel happens to be a different size.
  const ks = Object.keys(DIRPROTO);
  if(!ks.length) return null;
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for(const k of ks){
    x0 = Math.min(x0, DIRPROTO[k][0]); y0 = Math.min(y0, DIRPROTO[k][1]);
    x1 = Math.max(x1, DIRPROTO[k][0]); y1 = Math.max(y1, DIRPROTO[k][1]);
  }
  return {x0: x0 - DIRBOX_PAD, y0: y0 - DIRBOX_PAD, x1: x1 + DIRBOX_PAD, y1: y1 + DIRBOX_PAD};
}
function dirBoxHit(b, x, y){
  if(!b) return null;
  const t = 12;
  const nearL = Math.abs(x - b.x0) <= t, nearR = Math.abs(x - b.x1) <= t;
  const nearT = Math.abs(y - b.y0) <= t, nearB = Math.abs(y - b.y1) <= t;
  if(x < b.x0 - t || x > b.x1 + t || y < b.y0 - t || y > b.y1 + t) return null;
  if(nearT && nearL) return "dirnw"; if(nearT && nearR) return "dirne";
  if(nearB && nearL) return "dirsw"; if(nearB && nearR) return "dirse";
  if(nearL) return "dirleft"; if(nearR) return "dirright";
  if(nearT) return "dirtop";  if(nearB) return "dirbottom";
  return "dirmove";
}
pc.onmousedown = ev => {
  const r = pc.getBoundingClientRect();
  const x = (ev.clientX - r.left) / r.width * PW, y = (ev.clientY - r.top) / r.height * PH;
  // The direction grid moves as a GROUP.  Grabbing any of its nine ball positions - the
  // four that are also cells, and the five that are not - moves the whole 3x3, because
  // the decoder classifies the ball against all nine prototypes and moving one alone
  // would leave it with a grid that no longer matches the ball.
  for(const k in DIRPROTO){
    if(Math.hypot(DIRPROTO[k][0] - x, DIRPROTO[k][1] - y) < 12){
      dirDrag = {mode: "dirmove", x, y, box: dirBox(),
                 proto: JSON.parse(JSON.stringify(DIRPROTO))};
      ev.preventDefault(); return;
    }
  }
  const dbh = dirBoxHit(dirBox(), x, y);
  if(dbh && dbh !== "dirmove"){
    dirDrag = {mode: dbh, x, y, box: dirBox(),
               proto: JSON.parse(JSON.stringify(DIRPROTO))};
    ev.preventDefault(); return;
  }
  if(dbh === "dirmove"){
    dirDrag = {mode: "dirmove", x, y, box: dirBox(),
               proto: JSON.parse(JSON.stringify(DIRPROTO))};
    ev.preventDefault(); return;
  }
  for(const k of NAMES){
    if(DIR.includes(k)) continue;              // direction cells move with the group
    if(Math.hypot(CELLS[k][0]-x, CELLS[k][1]-y) < 12){ drag = k; ev.preventDefault(); return; }
  }
};
pc.onmousemove = ev => {
  if(dirDrag){
    const r = pc.getBoundingClientRect();
    const x = (ev.clientX - r.left) / r.width * PW, y = (ev.clientY - r.top) / r.height * PH;
    const b = dirDrag.box, m = dirDrag.mode;
    if(m === "dirmove"){
      const dx = x - dirDrag.x, dy = y - dirDrag.y;
      for(const k in DIRPROTO){ DIRPROTO[k] = [DIRPROTO[k][0] + dx, DIRPROTO[k][1] + dy]; }
      dirDrag.x = x; dirDrag.y = y;           // incremental, so no rounding drift
    } else {
      // Stretch: the box edges move, and the nine points follow proportionally - the grid
      // keeps its shape instead of being distorted point by point.
      let x0 = b.x0, y0 = b.y0, x1 = b.x1, y1 = b.y1;
      if(m === "dirleft" || m === "dirnw" || m === "dirsw") x0 = x;
      if(m === "dirright" || m === "dirne" || m === "dirse") x1 = x;
      if(m === "dirtop" || m === "dirnw" || m === "dirne") y0 = y;
      if(m === "dirbottom" || m === "dirsw" || m === "dirse") y1 = y;
      if(x1 - x0 > 8 && y1 - y0 > 8){
        const sx = (x1 - x0) / (b.x1 - b.x0), sy = (y1 - y0) / (b.y1 - b.y0);
        for(const k in dirDrag.proto){
          const p = dirDrag.proto[k];
          DIRPROTO[k] = [x0 + (p[0] - b.x0) * sx, y0 + (p[1] - b.y0) * sy];
        }
      }
    }
    for(const k of DIR){
      if(CELLS[k]) CELLS[k] = [Math.round(DIRPROTO[k][0]), Math.round(DIRPROTO[k][1])];
    }
    draw();
    return;
  }
  if(!drag) return;
  const r = pc.getBoundingClientRect();
  CELLS[drag] = [Math.round((ev.clientX - r.left) / r.width * PW),
                 Math.round((ev.clientY - r.top) / r.height * PH)];
  draw();
};
let justDragged = false;
pc.onmouseup = () => {
  if(dirDrag){
    // Same guard as a single-marker drag: the click that follows this mouseup would
    // otherwise add a cell.  Leaving it out here was the second time this bug appeared.
    dirDrag = null; justDragged = true;
    setTimeout(()=>{ justDragged = false; }, 0);
    postCells();
    document.getElementById('status').textContent = t.dirMoved;
    return;
  }
  if(drag){ const k = drag; drag = null; justDragged = true;
    setTimeout(()=>{ justDragged = false; }, 0);   // cleared after the click that follows
    postCells();
    document.getElementById('status').textContent = k + " → " + JSON.stringify(CELLS[k]); }
};
pc.onclick = ev => {
  // Browsers fire click AFTER mouseup, so a `drag` test here is always false by the
  // time it runs - which is how a drag also created a new cell.  The real guard is
  // positional: never add a cell where one already is.
  if(justDragged) return;
  // The detected list came up short (a keycap that lights up with no cell shows as
  // "a key is pressed and nothing reacts"), and the person watching knows where it
  // is better than any sampling threshold does.  Click the magnified panel to add.
  const r = pc.getBoundingClientRect();
  const x = (ev.clientX - r.left) / r.width * PW;
  const y = (ev.clientY - r.top) / r.height * PH;
  // Any existing marker, INCLUDING the five ball positions that are not cells
  // (neutral and the four diagonals): grabbing the grid by a diagonal and releasing
  // there used to be far enough from every cell to slip past this test.
  for(const k of NAMES){
    if(Math.hypot(CELLS[k][0]-x, CELLS[k][1]-y) < 14) return;
  }
  for(const k in DIRPROTO){
    if(Math.hypot(DIRPROTO[k][0]-x, DIRPROTO[k][1]-y) < 14) return;
  }
  let i = 1; while(CELLS["x"+i]) i++;
  const name = "x"+i;
  CELLS[name] = [Math.round(x), Math.round(y)];
  syncCells(); postCells(); buildTable();
  document.getElementById('status').textContent = t.addCell.replace("%s", name);
};

let rowRefs = {};
function buildTable(){
  // Structure once.  Rebuilding innerHTML on a timer - which is what this did -
  // destroys the <select> elements, so an open dropdown closes by itself within
  // half a second.  Only updateTable() runs on a timer now; the selects, buttons
  // and rows persist, and that also keeps focus and the open menu.
  const rows = [`<tr><th>${t.num}</th><th>${t.cell}</th><th>${t.key}</th>
                 <th>${t.state}</th><th>${t.stat}</th><th></th></tr>`];
  for(const k of NAMES){
    const opts = ["(ignore)"].concat(KEYS).map(x =>
      `<option value="${x}" ${mapping[k]===x?"selected":""}>${keyLabel(x)}</option>`).join("");
    rows.push(`<tr data-row="${k}"><td>${NUM[k]}</td><td>${k}</td>
      <td><select data-cell="${k}">${opts}</select></td>
      <td data-state="${k}"></td>
      <td data-stat="${k}"></td>
      <td><button data-jump="${k}:-1">${t.prevPress}</button>
          <button data-jump="${k}:1">${t.nextPress}</button>
          <button data-del="${k}" title="${t.del}">✕</button></td></tr>`);
  }
  document.getElementById('map').innerHTML = rows.join("");
  rowRefs = {};
  for(const k of NAMES){
    rowRefs[k] = {row: document.querySelector(`tr[data-row="${k}"]`),
                  state: document.querySelector(`[data-state="${k}"]`),
                  stat: document.querySelector(`[data-stat="${k}"]`)};
  }
  document.querySelectorAll('select[data-cell]').forEach(sel =>
    sel.onchange = () => { mapping[sel.dataset.cell] = sel.value; selected = sel.dataset.cell;
      fetch('/api/mapping',{method:'POST',body:JSON.stringify(mappingPayload())});
      updateTable(); });                      // NOT buildTable: that closed the menu
  document.querySelectorAll('button[data-jump]').forEach(btn =>
    btn.onclick = () => { const [c,d] = btn.dataset.jump.split(':');
      selected = c; jump(c, parseInt(d,10)); updateTable(); });
  document.querySelectorAll('button[data-del]').forEach(btn =>
    btn.onclick = () => { const c = btn.dataset.del;
      if(!confirm(t.deleted.replace("%s", c) + " ?")) return;
      delete CELLS[c]; delete stat[c]; syncCells(); postCells(); buildTable();
      document.getElementById('status').textContent = t.deleted.replace("%s", c); });
  updateTable();
}

function updateTable(){
  const pn = document.getElementById('panelinfo');
  if(pn) pn.textContent = t.panelNow.replace("%d", PANEL[0]).replace("%d", PANEL[1])
                                 .replace("%d", PW).replace("%d", PH);
  for(const k of NAMES){
    const r = rowRefs[k]; if(!r) continue;
    const s = stat[k];
    if(r.state){ r.state.textContent = s.down ? t.down : t.up;
                 r.state.className = s.down ? "down" : "unknown"; }
    if(r.stat){ r.stat.textContent = known[k] ? t.presses.replace("%d", s.n) : t.need; }
    if(r.row){ r.row.className = (k === selected) ? "sel" : ""; }
  }
}

function applyLang(){
  t = S[lang];
  document.title = t.title;
  updateTable();
  for(const [id,key] of [["h","h"],["b-start","start"],["b-back","back"],["b-prev","prev"],
      ["b-next","next"],["b-fwd","fwd"],["b-save","save"],["b-scan","scan"],
      ["b-discover","discover"],
      ["l-speed","speed"],
      ["l-lang","lang"],["l-zoom","zoom"],["l-panel","panel"],["l-table","table"],
      ["hint","hint"]])
    document.getElementById(id).textContent = t[key];
  document.getElementById('b-play').textContent = v.paused ? t.play : t.pause;
  buildTable();
}

// Redrawing the box from scratch is Shift+drag on the frame.  It used to be a button,
// which is redundant now that the box can be dragged and resized directly - but the
// capability stays, because a box that is somehow wrong must still be replaceable.
// The panel box is directly manipulable: drag inside it to move it, drag an edge or a
// corner to resize.  A crop window cannot be right if the only way to adjust it is to
// redraw it from scratch, and the box decides whether every cell sits on its keycap.
const BOX_TOL = 14;                     // frame pixels - 8 was too small to hit reliably
function boxHit(x, y){
  const x0 = PANEL[0], y0 = PANEL[1], x1 = x0 + PW, y1 = y0 + PH;
  const nearL = Math.abs(x - x0) <= BOX_TOL, nearR = Math.abs(x - x1) <= BOX_TOL;
  const nearT = Math.abs(y - y0) <= BOX_TOL, nearB = Math.abs(y - y1) <= BOX_TOL;
  if(x < x0 - BOX_TOL || x > x1 + BOX_TOL || y < y0 - BOX_TOL || y > y1 + BOX_TOL) return null;
  if(nearT && nearL) return "nw"; if(nearT && nearR) return "ne";
  if(nearB && nearL) return "sw"; if(nearB && nearR) return "se";
  if(nearL) return "left";  if(nearR) return "right";
  if(nearT) return "top";   if(nearB) return "bottom";
  return "move";
}
function cursorFor(h){
  if(!h) return panelMode ? "crosshair" : "default";
  if(h === "move") return "move";
  if(h === "left" || h === "right") return "ew-resize";
  if(h === "top" || h === "bottom") return "ns-resize";
  return (h === "nw" || h === "se") ? "nwse-resize" : "nesw-resize";
}
function frameXY(ev){
  const r = cv.getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width * VW, (ev.clientY - r.top) / r.height * VH];
}
let boxDrag = null;
cv.onmousedown = ev => {
  const [x, y] = frameXY(ev);
  if(ev.shiftKey){                       // Shift+drag redraws the box from scratch
    panelMode = true; boxDrag = null;
    panelRect = {x0: x, y0: y, x1: x, y1: y};
    ev.preventDefault();
    return;
  }
  const h = boxHit(x, y);
  if(h){
    boxDrag = {mode: h, x, y, box: [PANEL[0], PANEL[1], PW, PH]};
    ev.preventDefault();
    return;
  }
  if(panelMode){
    panelRect = {x0: x, y0: y, x1: x, y1: y};
    ev.preventDefault();
  }
};
cv.onmousemove = ev => {
  const [x, y] = frameXY(ev);
  if(boxDrag){
    const b = boxDrag.box, dx = x - boxDrag.x, dy = y - boxDrag.y;
    let x0 = b[0], y0 = b[1], x1 = b[0] + b[2], y1 = b[1] + b[3];
    // Explicitly, not by substring: "nw"/"se"/"ne"/"sw" contain no 'l', 'r' or 'b', so
    // indexOf() made every CORNER a no-op - which is why dragging a corner appeared to
    // do nothing at all.
    const m = boxDrag.mode;
    if(m === "move"){ x0 += dx; y0 += dy; x1 += dx; y1 += dy; }
    else {
      if(m === "left" || m === "nw" || m === "sw") x0 += dx;
      if(m === "right" || m === "ne" || m === "se") x1 += dx;
      if(m === "top" || m === "nw" || m === "ne") y0 += dy;
      if(m === "bottom" || m === "sw" || m === "se") y1 += dy;
    }
    x0 = Math.max(0, Math.min(x0, VW - 10)); y0 = Math.max(0, Math.min(y0, VH - 10));
    x1 = Math.max(x0 + 40, Math.min(x1, VW)); y1 = Math.max(y0 + 30, Math.min(y1, VH));
    panelRect = {x0, y0, x1, y1};
    ev.preventDefault();
    return;
  }
  const hov = boxHit(x, y);
  cv.style.cursor = cursorFor(hov);
  const hb = document.getElementById('handle');
  if(hb) hb.textContent = hov ? ("handle: " + hov) : "";
  if(panelMode && panelRect){
    panelRect.x1 = x; panelRect.y1 = y;
  }
};
cv.onmouseup = () => {
  if(boxDrag){
    if(panelRect){
      applyPanel(Math.round(panelRect.x0), Math.round(panelRect.y0),
                 Math.round(panelRect.x1 - panelRect.x0),
                 Math.round(panelRect.y1 - panelRect.y0));
    }
    boxDrag = null; panelRect = null;
    return;
  }
  if(!panelMode || !panelRect) return;
  const x0 = Math.max(0, Math.round(Math.min(panelRect.x0, panelRect.x1)));
  const y0 = Math.max(0, Math.round(Math.min(panelRect.y0, panelRect.y1)));
  const x1 = Math.min(VW, Math.round(Math.max(panelRect.x0, panelRect.x1)));
  const y1 = Math.min(VH, Math.round(Math.max(panelRect.y0, panelRect.y1)));
  panelRect = null; panelMode = false;
  if(x1 - x0 < 40 || y1 - y0 < 30){
    document.getElementById('status').textContent = t.panelHint; return;
  }
  applyPanel(x0, y0, x1 - x0, y1 - y0);
};
window.addEventListener('keydown', ev => {
  if(ev.key === "Escape" && panelMode){ panelMode = false; panelRect = null;
    document.getElementById('status').textContent = ""; return; }
  // Never hijack keys while a control has focus: arrow keys inside the mapping
  // dropdown choose an option, and stealing them would also seek the video.
  const tag = (ev.target && ev.target.tagName || "").toLowerCase();
  if(tag === "select" || tag === "input" || tag === "textarea" || ev.target.isContentEditable) return;
  const big = 5, oneFrame = 1/30;
  switch(ev.key){
    case " ": case "Spacebar": ev.preventDefault();
      v.paused ? v.play() : v.pause(); break;
    case "ArrowLeft": ev.preventDefault();
      v.currentTime = Math.max(0, v.currentTime - (ev.shiftKey ? oneFrame : big)); break;
    case "ArrowRight": ev.preventDefault();
      v.currentTime = Math.min(v.duration, v.currentTime + (ev.shiftKey ? oneFrame : big)); break;
    case ",": case "<": ev.preventDefault(); step(-oneFrame); break;
    case ".": case ">": ev.preventDefault(); step(oneFrame); break;
    case "Home": ev.preventDefault(); v.currentTime = 0; break;
    case "End": ev.preventDefault(); v.currentTime = Math.max(0, v.duration - 0.1); break;
  }
});
document.getElementById('lang').value = lang;
document.getElementById('lang').onchange = e => { lang = e.target.value; applyLang(); };
document.getElementById('b-play').onclick = () => { v.paused ? v.play() : v.pause(); };
document.getElementById('b-prev').onclick = () => step(-1/30);
document.getElementById('b-next').onclick = () => step(1/30);
document.getElementById('b-back').onclick = () => { v.currentTime -= 5; };
document.getElementById('b-fwd').onclick = () => { v.currentTime += 5; };
document.getElementById('b-start').onclick = () => { v.currentTime = 0; };
function mappingPayload(){
  // Send EVERY cell, always.  The payload used to be the `mapping` object, which only gained
  // an entry when its dropdown was changed, and the server REPLACES the file with what it
  // receives - so a cell nobody had touched was missing from the payload and the next
  // unrelated change deleted its assignment.  That is how video 1 lost x6 -> item.
  const out = {};
  for(const k of NAMES){
    const s = document.querySelector(`select[data-cell="${k}"]`);
    out[k] = s ? s.value : (mapping[k] || "(ignore)");
  }
  return out;
}
document.getElementById('b-save').onclick = () =>
  fetch('/api/mapping',{method:'POST',body:JSON.stringify(mappingPayload())})
    .then(()=>document.getElementById('status').textContent =
      t.saved + " (" + NAMES.length + " cells)");
document.getElementById('speed').onchange = e => { v.playbackRate = parseFloat(e.target.value); };
document.getElementById('zoom').onchange = e => { const z = parseFloat(e.target.value);
  cv.style.width = (VW*z) + "px"; cv.style.height = (VH*z) + "px"; };
function fitZoom(){
  // A 1920 px wide canvas next to the table is unusable, so the default is whatever
  // comes out near 900 px wide, rounded down to a choice the menu actually offers.
  const menu = Array.from(document.getElementById('zoom').options).map(o => parseFloat(o.value));
  const want = Math.min(1, 900 / VW);
  return menu.filter(z => z <= want + 1e-6).pop() || menu[0];
}
v.addEventListener('loadedmetadata', () => {
  VW = v.videoWidth || 1280; VH = v.videoHeight || 720;
  cv.width = VW; cv.height = VH;                 // native space, CSS-only scaling
  const z = fitZoom();
  document.getElementById('zoom').value = String(z);
  cv.style.width = (VW*z) + "px"; cv.style.height = (VH*z) + "px";
  document.getElementById('status').textContent = "frame " + VW + "x" + VH;
});
function step(dt){ v.pause(); v.currentTime = Math.max(0, v.currentTime + dt); }
function jump(cell, dir){
  // "where is this key pressed" - the identification workhorse.  Only intervals
  // seen so far are known, which is what the scan button is for.
  const ev = (events[cell]||[]).slice().sort((a,b)=>a[0]-b[0]);
  const now = v.currentTime;
  let target = null;
  if(dir > 0){ const nxt = ev.find(e => e[0] > now + 0.05); if(nxt) target = nxt[0]; }
  else { const prv = ev.filter(e => e[0] < now - 0.05); if(prv.length) target = prv[prv.length-1][0]; }
  const st = document.getElementById('status');
  if(target === null){ st.textContent = (dir>0? t.noNext : t.noPrev).replace("%s", cell); return; }
  v.pause();
  v.currentTime = Math.max(0, target - 0.6);
  st.textContent = cell + " " + t.jumpTo.replace("%.2f", target.toFixed(2));
}
function seekTo(t){
  return new Promise(res => {
    if(Math.abs(v.currentTime - t) < 0.002){ res(); return; }
    const on = () => { v.removeEventListener('seeked', on); res(); };
    v.addEventListener('seeked', on);
    v.currentTime = t;
  });
}
let discovering = false;
document.getElementById('b-discover').onclick = async () => {
  const btn = document.getElementById('b-discover');
  if(discovering){                       // pressing it again stops it
    discovering = false; btn.textContent = t.discover; return;
  }
  btn.textContent = t.scanStop;
  // Positions, not presses: sample one frame every few seconds and collect the
  // centroids of keycap-sized bright blobs.  A keycap that is never pressed cannot
  // be found this way, but one that lights up even twice is enough to locate it -
  // which is the gap that hand-reading the screenshot left behind.
  if(discovering) return;
  discovering = true;
  v.pause();
  // 1 s steps: at 3 s a keycap held for half a second is missed 5 times out of 6, and
  // the rare keycaps are exactly the ones this button exists to find.
  const st = document.getElementById('status'), step = 1, pts = [];
  // From the CURRENT frame, like the playback scan: once the keycaps you care about have
  // been found there is nothing to gain from re-walking the start of the video, and the
  // points already collected are not thrown away either (each run only adds cells for
  // positions that no existing cell covers).
  for(let t = v.currentTime; t < v.duration - 0.05 && discovering; t += step){
    await seekTo(t);
    for(const b of components()){
      if(b.area < 150 || b.area > 1600) continue;
      if(b.cx < 110) continue;            // the 3x3 direction grid is already measured
      pts.push([b.cx, b.cy]);
    }
    st.textContent = S[lang].discovering.replace("%d", Math.round(100 * t / v.duration));
  }
  // Cancelling must not throw away the work: the points already collected are
  // processed exactly as if the scan had finished early.
  const stopped = !discovering;
  const clusters = [];
  for(const [x, y] of pts){
    let hit = null;
    for(const c of clusters) if(Math.hypot(c.x/c.n - x, c.y/c.n - y) < 10){ hit = c; break; }
    if(hit){ hit.x += x; hit.y += y; hit.n++; } else clusters.push({x, y, n: 1});
  }
  let added = 0;
  for(const c of clusters){
    // No repetition filter: a 150-1600 px blob cannot appear by accident, and a keycap
    // that lights up ONCE in 43 minutes is exactly the one that hand-reading missed.
    if(c.n < 1) continue;
    const cx = c.x/c.n, cy = c.y/c.n;
    if(NAMES.some(k => Math.hypot(CELLS[k][0]-cx, CELLS[k][1]-cy) < 14)) continue;
    let i = 1; while(CELLS["x"+i]) i++;
    CELLS["x"+i] = [Math.round(cx), Math.round(cy)];
    added++;
  }
  discovering = false;
  document.getElementById('b-discover').textContent = t.discover;
  syncCells(); postCells(); buildTable();
  // Report the numbers at every stage.  "Nothing happened" is not a symptom one can act
  // on: 0 points means the frame was not decoded, points but 0 clusters means the
  // clustering, clusters but 0 added means everything is already covered.
  const summary = "found " + pts.length + " bright spots -> " + clusters.length
                + " positions -> " + added + " cells added (now " + NAMES.length + ")";
  st.textContent = (stopped ? S[lang].discoverStopped.replace("%d", added)
                  : added ? S[lang].discoverDone.replace("%d", clusters.length).replace("%d", added)
                          : S[lang].discoverNone) + " | " + summary;
};
let scanning = false;
document.getElementById('b-scan').onclick = async () => {
  const btn = document.getElementById('b-scan');
  if(scanning){                          // pressing it again stops it
    scanning = false; btn.textContent = t.scan; return;
  }
  scanning = true;
  btn.textContent = t.scanStop;
  // Scan FORWARD FROM HERE, keeping what is already known: once every key has been
  // identified there is nothing to gain from re-walking the first 40 minutes, and
  // throwing the collected intervals away each time was pure loss.
  const st = document.getElementById('status'), rate = v.playbackRate;
  v.playbackRate = 8;                     // ~5.5 min for a full 43 min video
  await v.play();
  const tick = () => {
    st.textContent = t.scanning.replace("%d", Math.round(100*v.currentTime/v.duration));
    if(scanning && !v.ended && v.currentTime < v.duration - 0.05) setTimeout(tick, 300);
    else { scanning = false; document.getElementById('b-scan').textContent = t.scan;
           v.pause(); v.playbackRate = rate;
           st.textContent = scanning ? t.scanStopped : t.scanDone; updateTable(); }
  };
  tick();
};
// No separate stop button: each scan's own button becomes 停止 while it runs.
v.onplay = () => document.getElementById('b-play').textContent = t.pause;
v.onpause = () => document.getElementById('b-play').textContent = t.play;
const seek = document.getElementById('seek');
seek.oninput = () => { v.currentTime = seek.value/1000 * v.duration; };
function loop(){
  if(v.readyState >= 2){
    if(v.duration && !isNaN(v.duration)) seek.value = 1000 * v.currentTime / v.duration;
    measure(); draw();
    document.getElementById('pos').textContent =
      "t=" + v.currentTime.toFixed(2) + "s / " + v.duration.toFixed(1) + "s";
  }
  requestAnimationFrame(loop);
}
let lastTable = 0;
setInterval(updateTable, 300);   // text only: never rebuild the selects
// The magnified view is derived from the panel box, never hardcoded: with a 330x170
// panel (a 1080p video) a fixed 600x276 canvas stretches the crop AND puts every marker
// at the wrong place.  This has to run AFTER the canvas is looked up - assigning it here
// rather than next to the panel variables is deliberate, because `const pc` is declared
// further down and touching it earlier throws a TDZ ReferenceError, which kills the
// whole script (the page then shows nothing but its HTML skeleton).
pc.width = PW * 2; pc.height = PH * 2;
applyLang(); loop();
</script></body></html>
"""


class Server(BaseHTTPRequestHandler):
    video: Path
    mapping_path: Path
    mapping: dict = {}

    def log_message(self, *_a: object) -> None:      # keep the console usable
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                        # noqa: N802
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            html = (PAGE
                    .replace("__CELLS__", json.dumps(self.cells or DEFAULT_CELLS))
                    .replace("__PANEL__", json.dumps(self.panel))
                    .replace("__PW__", str(self.panel_size[0]))
                    .replace("__PH__", str(self.panel_size[1]))
                    .replace("__KEYS__", json.dumps(self.keys))
                    .replace("__MAPPING__", json.dumps(self.mapping))
                    .replace("__DIRPROTO__", json.dumps(self.dir_proto))
                    .replace("__H__", "速通按键标注").replace("__TITLE__", "速通按键标注"))
            self._send(html.encode(), "text/html; charset=utf-8")
            return
        if u.path == "/video.mp4":
            self._send_video()
            return
        self._send(b"not found", "text/plain", 404)

    def _send_video(self) -> None:
        """Plain file serving with Range support (the browser needs it to seek)."""
        size = self.video.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            part = rng.split("=", 1)[1].split(",")[0]
            a, _, b = part.partition("-")
            start = int(a) if a else 0
            end = int(b) if b else size - 1
        end = min(end, size - 1)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if rng:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.end_headers()
        with open(self.video, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


    def _sync_pack(self) -> None:
        """Keep data/videos/<stem>/meta.json in step with whatever was just saved.

        The pack is the layout people and tools are meant to read, so if the page wrote only the
        legacy files it would go stale and start lying.  Writing both is deliberate during the
        transition; a stale pack would be worse than no pack.
        """
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from ash.data.video_pack import load_meta, save_meta
            meta = load_meta(type(self)._vp.stem_of(video)) or {"id": type(self)._vp.stem_of(video)}
            meta.update({"id": type(self)._vp.stem_of(video),
                         "video": type(self).video.name,
                         "panel": list(type(self).panel),
                         "panel_size": list(type(self).panel_size),
                         "cells": {k: list(v) for k, v in type(self).cells.items()},
                         "dir_proto": {k: list(v) for k, v in (type(self).dir_proto or {}).items()},
                         "mapping": dict(type(self).mapping)})
            save_meta(type(self)._vp.stem_of(video), meta)
        except Exception as exc:                      # noqa: BLE001
            print("could not update the video pack: %s" % exc)

    def do_POST(self) -> None:                       # noqa: N802
        if urlparse(self.path).path == "/api/cells":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
                cells = {str(k): [int(v[0]), int(v[1])]
                         for k, v in (data.get("cells") or {}).items()}
                panel = [int(v) for v in (data.get("panel") or [])]
                psize = [int(v) for v in (data.get("panel_size") or [])]
                proto = {str(k): [int(round(v[0])), int(round(v[1]))]
                         for k, v in (data.get("dir_proto") or {}).items()}
            except (ValueError, TypeError, IndexError):
                self._send(b"bad json", "text/plain", 400)
                return
            if len(panel) == 2 and len(psize) == 2 and psize[0] > 20 and psize[1] > 20:
                # The box is the user's, measured on the frame - accept it, it is the
                # only thing that fixes a layout whose crop was wrong.
                type(self).panel = panel
                type(self).panel_size = psize
            if proto:
                type(self).dir_proto = proto      # scaled with the box, so it must be saved
            type(self).cells = cells
            self._sync_pack()
            blob = load_json(self.cells_path)          # keep dir_proto and anything else
            blob.update({"video": self.video.name, "panel": type(self).panel,
                         "panel_size": type(self).panel_size, "cells": cells})
            if type(self).dir_proto:
                blob["dir_proto"] = type(self).dir_proto
            atomic_write(self.cells_path, blob)
            self._send(b'{"ok":true}', "application/json")
            return
        if urlparse(self.path).path != "/api/mapping":
            self._send(b"not found", "text/plain", 404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            self._send(b"bad json", "text/plain", 400)
            return
        # Filter against the CURRENT cells, not the built-in list: filtering by
        # DEFAULT_CELLS silently dropped the mapping of every cell added by hand
        # (x1, and c7 before it) - reported as "saving loses the new keys".
        known = set(type(self).cells or DEFAULT_CELLS)
        keep = {k: v for k, v in data.items() if k in known}
        type(self).mapping = keep
        blob = load_json(self.mapping_path)            # keep the measured positions
        blob.update({"video": self.video.name, "cells": keep,
                     "positions": {k: list(v) for k, v in type(self).cells.items()}})
        atomic_write(self.mapping_path, blob)
        self._send(b'{"ok":true}', "application/json")

    keys: list[str] = []
    panel: list = list(PANEL)
    panel_size: list = [PANEL_W, PANEL_H]
    cells: dict = {}
    cells_path: Path = Path("runs/cells.json")
    dir_proto: dict = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    video = Path(args.video).resolve()
    mapping_path = Path(args.mapping).resolve() if args.mapping else \
        _vp.meta_path(_vp.stem_of(video))

    try:
        from ash.actions.space import BUTTONS as _B
        keys = list(_B)
    except Exception:                                 # noqa: BLE001
        keys = ["up", "down", "left", "right", "jump", "attack", "dash", "special",
                "interact", "menu", "ult", "weapon_switch", "cancel", "item"]

    cells_path = _vp.meta_path(_vp.stem_of(video))
    # Prefer the unified per-video pack (data/videos/<stem>/meta.json); the legacy pair of
    # cells/mapping files is the fallback so unmigrated videos keep working.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from ash.data.video_pack import load_meta as _load_meta
        _meta = _load_meta(_vp.stem_of(video))
    except Exception:                                 # noqa: BLE001
        _meta = None
    if _meta:
        if _meta.get("panel") and _meta.get("panel_size"):
            Server.panel = [int(v) for v in _meta["panel"]]
            Server.panel_size = [int(v) for v in _meta["panel_size"]]
            print("panel: %s size %s from the video pack" % (Server.panel, Server.panel_size))
        if _meta.get("cells"):
            Server.cells = {k: list(v) for k, v in _meta["cells"].items()}
            print("cells: %d loaded from the video pack" % len(Server.cells))
        if _meta.get("dir_proto"):
            Server.dir_proto = {k: list(v) for k, v in _meta["dir_proto"].items()}
        if _meta.get("mapping"):
            Server.mapping.update({k: v for k, v in _meta["mapping"].items()})
        if _meta.get("style"):
            Server.style = _meta["style"]
    if cells_path.exists() and not Server.cells:
        try:
            blob = json.loads(cells_path.read_text())
        except Exception as exc:                      # noqa: BLE001
            # Do NOT fall back silently.  That is what hid this: an unparseable cell
            # list quietly became the built-in ten, the page then saved ITS idea of
            # the cells back over the real file, and a whole session of identifying
            # keys and directions was lost without an error anywhere.
            print("FATAL: %s exists but cannot be parsed: %s" % (cells_path, exc))
            print("Refusing to start with a different cell list than the one on disk.")
            return 2
        got = blob.get("cells") or {}
        if got:
            Server.cells = {k: list(v) for k, v in got.items()}
            print("cells: %d loaded from %s" % (len(Server.cells), cells_path))
        if blob.get("panel") and blob.get("panel_size"):
            # The box is per video and saved with the cells.  Without this the server
            # falls back to the 720p default, which for a 1080p video is the middle of the
            # game - and nothing fails loudly: the page just crops the wrong region, the
            # same symptom as a wrong box.
            Server.panel = [int(v) for v in blob["panel"]]
            Server.panel_size = [int(v) for v in blob["panel_size"]]
            print("panel: %s size %s from %s" % (Server.panel, Server.panel_size, cells_path))
        if blob.get("dir_proto"):
            Server.dir_proto = {k: list(v) for k, v in blob["dir_proto"].items()}
            print("directions: %d positions (3x3 grid)" % len(Server.dir_proto))
        else:
            print("no dir_proto in %s: directions are read from the four keycaps "
                  "(this video draws the arrow keys, not a ball)" % cells_path)
    Server.cells_path = cells_path
    Server.video = video
    Server.mapping_path = mapping_path
    Server.keys = keys
    Server.mapping = {k: "(ignore)" for k in (set(DEFAULT_CELLS) | set(Server.cells or {}))}
    # L/R were pinned offline by camera shift + occupation; seed them.
    # All four directions have unambiguous defaults; only the action buttons need a human
    # to look at the panel and decide.  Seeding just L/R left U/D unassigned, which reads
    # as "not identified" for keys nobody has to identify.
    Server.mapping.update({"L": "left", "R": "right", "U": "up", "D": "down"})
    if mapping_path.exists():
        try:
            got = json.loads(mapping_path.read_text()).get("cells") or {}
            # Accept every cell THIS video has, not just the built-in layout: DEFAULT_CELLS
            # knows only L/R/U/D/c1..c6, so filtering by it silently discarded the mapping of
            # every cell added by hand (x1..x6) - on every single restart.  The save path was
            # fixed for this and the load path was missed, which is why it kept coming back.
            known = set(Server.cells or DEFAULT_CELLS) | set(got)
            Server.mapping.update({k: v for k, v in got.items() if k in known})
        except Exception:                             # noqa: BLE001
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Server)
    url = "http://127.0.0.1:%d/" % args.port
    print("serving %s" % video.name)
    print("open: %s" % url)
    print("mapping: %s" % mapping_path)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
