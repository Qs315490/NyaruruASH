"""Turn the speedrun videos' keycast into action labels, aligned to the corpus grid.

The whole route exists to get here: the overlay in these videos is the only source of
action labels for them, and the page established which keycap is which.  This is the
deterministic extraction the labels need - one pass, no browser, no adaptive
thresholds (measured: a pressed keycap reads 205-240, the panel about 55, so 150
separates them, and a keycap is 150-1600 px while a speck or a piece of overlay
furniture is not).

Output, per tick of the CONTROL grid (config/game.yaml: control_interval_s = 0.25 s,
the same 4 fps the corpus frames use, so the two line up):

    held     keys held AT the tick instant   (what a policy imitating the frame sees)
    tapped   keys pressed at any point INSIDE the tick window
             (a 0.1 s tap falls between 4 fps samples; dropping it would silently
              lose input, so it is recorded rather than smoothed away)
    masks    action-space index for `held`, or -1 when the combination is not
             expressible - our space has right/left variants and down, but no
             diagonals like (up, right), and that gap is reported, not hidden

    uv run python scripts/extract_keycast_labels.py --video data/video-src/BV19s4y1y7un.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ash.data.video_pack import artifact_path as _ap  # noqa: E402

from ash.actions.space import ActionSpace, BUTTONS  # noqa: E402

PANEL_W, PANEL_H = 300, 138
KEY_MIN, KEY_MAX = 150, 1600          # for a 720p panel; rescaled from the panel size below
BRIGHT = 150
DIR_LABELS = {"U": ["up"], "UL": ["up", "left"], "UR": ["up", "right"],
              "L": ["left"], "-": [], "R": ["right"],
              "DL": ["down", "left"], "DR": ["down", "right"], "D": ["down"]}


def load(path: str | None, default: dict) -> dict:
    if path is None:
        return default
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:                          # noqa: BLE001
        print("could not read %s (%s); using the default" % (path, exc))
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--fps", type=int, default=30, help="decode rate for the panel")
    ap.add_argument("--tick", type=float, default=0.25, help="control interval, seconds")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=float, default=0.0, help="seconds, 0 = whole video")
    args = ap.parse_args()

    video = Path(args.video)
    stem = stem_of(video)
    # Default to the per-video files the page writes.  Without this the layout is not
    # found at all unless the caller passes the paths, and the next video silently
    # produced "no cells" (the first video worked only because I named them explicitly).
    cells_blob = load(args.cells or str(vp.artifact_path(stem, "cells")),
                      {"cells": {}, "dir_proto": {}, "panel": [250, 582]})
    cells = {k: tuple(v) for k, v in cells_blob.get("cells", {}).items()}
    proto = {k: tuple(v) for k, v in cells_blob.get("dir_proto", {}).items()}
    mapping_blob = load(args.mapping or str(vp.artifact_path(stem, "mapping")),
                        {"cells": {}})
    mapping = {k: v for k, v in mapping_blob.get("cells", {}).items()
               if v != "(ignore)" and k in cells}
    px, py = (int(v) for v in cells_blob.get("panel", [250, 582]))
    pw, ph = (int(v) for v in cells_blob.get("panel_size", [PANEL_W, PANEL_H]))
    # Rescale the keycap size limits from the panel width.  Left at the 720p values they
    # reject every keycap on a 1080p video (a keycap there is ~1600-2000 px), which shows
    # up as "the video has no labels" rather than as an error.
    # Two keycast styles exist in the corpus.  "bright" (the ball widget) lights the
    # pressed keycap UP; "bluekey" (the keyboard widget) paints it BLUE, so its luminance
    # goes DOWN - the brightness rule then fires on the idle keycaps and the labels are
    # nonsense (95% tapped, 61% unexpressible, measured on BV1hc411M7GW).
    style = str(cells_blob.get("style", "bright"))
    print("keycap style: %s" % style)
    _k = pw / 300.0
    key_min, key_max = KEY_MIN * _k * _k, KEY_MAX * _k * _k
    print("panel %dx%d -> keycap blob area %.0f..%.0f px" % (pw, ph, key_min, key_max))
    if not cells:
        print("no cells: pass --cells runs/keycast-cells-<video>.json")
        return 2
    print("cells %d | mapped %d (%s)"
          % (len(cells), len(mapping), ", ".join("%s=%s" % kv for kv in mapping.items())))
    use_ball = bool(proto)
    if not use_ball:
        print("no dir_proto: directions are decoded as ordinary keycaps "
              "(this video draws the arrow keys instead of a ball)")

    # cell -> button name position in BUTTONS (for building masks)
    bidx = {name: i for i, name in enumerate(BUTTONS)}
    btn_cells = {k: mapping[k] for k in cells
                 if k in mapping and k not in DIR_LABELS and mapping[k] in bidx}
    unknown_cells = [k for k in cells if k not in mapping]
    if unknown_cells:
        print("cells with no mapping (recorded as bits, not as actions): %s"
              % ", ".join(unknown_cells))
    space = ActionSpace.minimal()
    mask_to_idx = {mk: i for i, mk in enumerate(space.masks)}
    # Fallback for combinations the space cannot express (18% of this video's ticks:
    # direction+dash first, then the diagonals, then direction+special).  The raw bits
    # stay in held/tapped; `masks` gets the nearest expressible one, chosen by, in order:
    #   1. smallest Hamming distance (fewest bits added or dropped);
    #   2. on a tie, keep the BUTTON bits - a direction held here is usually still held
    #      on the next tick, whereas a dash/jump/attack is an event that is lost for
    #      good if it is dropped;
    #   3. on a tie, keep the horizontal direction (right 26.9% / left 23.6% against
    #      down 6.0% / up 3.6% - platformers move sideways);
    #   4. lowest mask index, so the choice is deterministic.
    button_bits = 0
    for nm in ("jump", "attack", "dash", "special", "ult", "weapon_switch",
               "item", "interact", "cancel", "menu"):
        if nm in bidx:
            button_bits |= 1 << bidx[nm]

    def nearest_mask(mk: int) -> int:
        best, best_key = 0, None
        for i, cand in enumerate(space.masks):
            miss = mk & ~cand
            key = (bin(mk ^ cand).count("1"),
                   bin(miss & button_bits).count("1"),
                   int(bool(mk & (1 << bidx["right"])) and not (cand & (1 << bidx["right"])))
                   + int(bool(mk & (1 << bidx["left"])) and not (cand & (1 << bidx["left"]))),
                   i)
            if best_key is None or key < best_key:
                best, best_key = cand, key
        return best

    # ---- one streaming pass -----------------------------------------------------
    cmd = ["ffmpeg", "-v", "error", "-i", str(video)]
    if args.limit:
        cmd += ["-t", str(args.limit)]
    # Crop and buffers follow the VIDEO's panel size, not the 720p default: a mismatch
    # (crop 375x85 into 300x138 buffers) silently decodes nothing at all, which reads as
    # "this video's keys never fire" rather than as an error.
    cmd += ["-vf", "fps=%d,crop=%d:%d:%d:%d" % (args.fps, pw, ph, px, py),
            "-fps_mode", "cfr",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n_px = pw * ph * 3
    states: list[np.ndarray] = []
    names = ["up", "down", "left", "right"] + sorted(btn_cells.values())
    names = list(dict.fromkeys(names))
    nidx = {n: i for i, n in enumerate(names)}
    while True:
        raw = proc.stdout.read(n_px)
        if not raw or len(raw) < n_px:
            break
        g = np.frombuffer(raw, np.uint8).reshape(ph, pw, 3)      # BGR
        if style == "bluekey":
            b = g[:, :, 0].astype(np.int16)
            m = ((b > 110) & (b - g[:, :, 2] > 40) & (b - g[:, :, 1] > 40)).astype(np.uint8)
        else:
            m = (g.mean(axis=2) > BRIGHT).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 4)
        keep = [j for j in range(1, n)
                if key_min <= stats[j, cv2.CC_STAT_AREA] <= key_max]
        row = np.zeros(len(names), bool)
        # directions: biggest kept blob left of x=110, classified to the 9 positions
        ball = None if not use_ball else None
        for j in ([] if not use_ball else keep):
            if cent[j][0] < 110 and (ball is None or stats[j, cv2.CC_STAT_AREA]
                                     > stats[ball, cv2.CC_STAT_AREA]):
                ball = j
        if ball is not None and proto:
            cx, cy = cent[ball]
            best = min(proto, key=lambda k: (proto[k][0] - cx) ** 2 + (proto[k][1] - cy) ** 2)
            for nm in DIR_LABELS.get(best, []):
                if nm in nidx:
                    row[nidx[nm]] = True
        # buttons: the cell centre must be covered by a kept blob
        # When there is no ball the direction cells are ordinary keycaps, and they must
        # carry ACTION names like btn_cells does - taking them straight from `cells` gave
        # coordinates where an action name was expected (KeyError on the tuple).
        dir_cells = ({} if use_ball else
                     {k: mapping[k] for k in cells
                      if k in DIR_LABELS and k in mapping and mapping[k] in bidx})
        for k, nm in list(btn_cells.items()) + list(dir_cells.items()):
            cx, cy = cells[k]
            on = False
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < ph and 0 <= xx < pw and int(lab[yy, xx]) in keep:
                        on = True
                        break
                if on:
                    break
            row[nidx[nm]] = on
        states.append(row)
    proc.wait()
    S = np.asarray(states, bool)
    print("decoded %d frames at %dfps (%.1f s)" % (len(S), args.fps, len(S) / args.fps))

    # ---- aggregate onto the control grid ---------------------------------------
    # Tick k covers frames [k * fps * tick, (k+1) * fps * tick) using FLOAT bounds.
    # Rounding the step to whole frames is a time-scale error: 30 fps * 0.25 s is 7.5
    # frames, and round() made it 8, so every tick lasted 0.2667 s and the whole label
    # timeline came out 6.7% (161 s) short of the video - and therefore misaligned
    # against the corpus, which is the one thing these labels exist to line up with.
    frame_step = args.fps * args.tick
    n_ticks = int(len(S) / frame_step)
    held = np.zeros((n_ticks, len(names)), bool)
    tapped = np.zeros((n_ticks, len(names)), bool)
    for k in range(n_ticks):
        a = int(round(k * frame_step))
        b = min(len(S), max(a + 1, int(round((k + 1) * frame_step))))
        win = S[a:b]
        held[k] = win[-1]
        tapped[k] = win.any(axis=0)
    masks = np.full(n_ticks, -1, np.int64)
    unsupported = 0
    for k in range(n_ticks):
        mk = 0
        for nm, i in nidx.items():
            if held[k, i]:
                mk |= 1 << bidx[nm]
        idx = mask_to_idx.get(mk)
        if idx is None:
            unsupported += 1
            masks[k] = mask_to_idx[nearest_mask(mk)]
        else:
            masks[k] = idx

    corpus = vp.artifact(stem, "corpus-4fps")
    corpus_frames = None
    if corpus.exists():
        corpus_frames = int(np.load(corpus, mmap_mode="r").shape[0])
    out = Path(args.out) if args.out else vp.artifact_path(stem, "labels")
    np.savez_compressed(out, times=np.arange(n_ticks) * args.tick, held=held,
                        tapped=tapped, masks=masks, names=np.array(names))
    print("\nwrote %s" % out)
    print("ticks %d (= %.1f s at %.2f s) | corpus frames %s"
          % (n_ticks, n_ticks * args.tick, args.tick, corpus_frames))
    if corpus_frames is not None:
        print("alignment: %+d frames (%.1f s) difference"
              % (n_ticks - corpus_frames, (n_ticks - corpus_frames) * args.tick))
    raw_empty = float((~held.any(axis=1)).mean())
    print("\nnoop in the truth     %.1f%% of ticks (nothing held at all)" % (100 * raw_empty))
    print("unsupported combos    %.1f%% of ticks mapped to the nearest expressible mask"
          " (raw bits kept in held/tapped)" % (100 * unsupported / max(1, n_ticks)))
    print("noop after mapping    %.1f%% of ticks" % (100 * (masks == 0).mean()))
    print("per-key held share:")
    for nm, i in sorted(nidx.items(), key=lambda kv: -held[:, kv[1]].mean()):
        t = tapped[:, i]
        print("   %-14s held %5.1f%%   tapped %5.1f%%   presses %d"
              % (nm, 100 * held[:, i].mean(), 100 * t.mean(),
                 int(np.sum(t & ~np.r_[False, t[:-1]]))))
    taps_between = int((tapped & ~held).sum())
    print("taps that fall BETWEEN 4 fps samples: %d (these would be lost by naive "
          "downsampling)" % taps_between)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
