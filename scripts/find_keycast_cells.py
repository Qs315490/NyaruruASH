"""Find every keycap of the keycast widget, including the rarely pressed ones.

The cell list has to be complete: a key that lights up with no cell pointing at it
shows as "a key is pressed but nothing in the table reacts", which is exactly what
was reported.  Hand-reading positions off a zoomed screenshot found ten and missed
the rest, so this finds them from the pixels:

  * a pressed keycap is a large bright disc (~30 px) on a dark blue panel;
  * sample many frames across the whole video (a rarely used key only appears a few
    times in 43 minutes, so a short sample misses it);
  * collect the centroids of all bright blobs, then find the peaks of that
    distribution - each real keycap is a tight cluster of centroids;
  * report the count and the positions, so "did we get them all" is a number
    rather than an impression.

The left widget is one ball at a few discrete positions, so its hits come out as
separate peaks too; they are kept and labelled as the direction group.

    uv run python scripts/find_keycast_cells.py --video data/video-src/BV19s4y1y7un.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ash.data.video_pack as vp  # noqa: E402
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ash.data.video_pack import artifact_path as _ap  # noqa: E402

PANEL = (250, 582)
PANEL_W, PANEL_H = 300, 138


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--panel", default="%d,%d" % PANEL)
    ap.add_argument("--panel-size", default="%d,%d" % (PANEL_W, PANEL_H),
                    help="panel width,height - 1080p videos have bigger keycaps")
    ap.add_argument("--min-area", type=int, default=200, help="pressed keycap area, px")
    ap.add_argument("--bright", type=int, default=170, help="pressed keycap mean value")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    px, py = (int(v) for v in args.panel.split(","))
    pw, ph = (int(v) for v in args.panel_size.split(","))
    out = Path(args.out) if args.out else \
        vp.meta_path(vp.stem_of(args.video))

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", args.video],
        capture_output=True, text=True, check=True).stdout.split()
    w_full, h_full, dur = int(float(probe[0])), int(float(probe[1])), float(probe[2])

    vote = np.zeros((ph, pw), np.float32)
    hits, frames = 0, 0
    for i in range(args.samples):
        t = dur * (i + 0.5) / args.samples
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "%.2f" % t, "-i", args.video,
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=True).stdout
        if len(raw) != w_full * h_full:
            continue
        g = np.frombuffer(raw, np.uint8).reshape(h_full, w_full)[py:py + ph, px:px + pw]
        frames += 1
        m = (g > 150).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        for j in range(1, n):
            a = stats[j, cv2.CC_STAT_AREA]
            if a < args.min_area:
                continue
            if float(g[lab == j].mean()) < args.bright:
                continue
            cxx, cyy = int(round(cent[j][0])), int(round(cent[j][1]))
            if 0 <= cyy < ph and 0 <= cxx < pw:
                vote[cyy, cxx] += 1
                hits += 1
    print("sampled %d frames | %d pressed-keycap hits | %.2f keys down per frame"
          % (frames, hits, hits / max(1, frames)))

    sm = cv2.GaussianBlur(vote, (0, 0), 3)
    mx = cv2.dilate(sm, np.ones((15, 15), np.uint8))
    ys, xs = np.nonzero((sm >= mx - 1e-9) & (sm > 0.02))
    order = np.argsort(-sm[ys, xs])
    cells: list[tuple[float, float, float]] = []
    for i in order:
        x, y = float(xs[i]), float(ys[i])
        if all((x - c[0]) ** 2 + (y - c[1]) ** 2 > 18 ** 2 for c in cells):
            cells.append((x, y, float(sm[int(y), int(x)])))
    cells.sort(key=lambda c: (round(c[1] / 16), c[0]))
    print("%d cells:" % len(cells))
    for i, (x, y, w) in enumerate(cells):
        kind = "dir" if x < 110 else "btn"
        print("   %2d  x=%6.1f y=%6.1f  weight=%5.2f  %s" % (i + 1, x, y, w, kind))

    # name them: the four direction peaks by geometry, the rest c1..cN by position
    dirs = [(x, y, w) for x, y, w in cells if x < 110]
    btns = [(x, y, w) for x, y, w in cells if x >= 110]
    named: dict[str, list[float]] = {}
    used: set[int] = set()
    for label, want in (("L", -1), ("R", 1)):
        cxs = [(i, c) for i, c in enumerate(dirs)]
        pick = max(cxs, key=lambda ic: want * ic[1][0]) if cxs else None
        if pick and pick[0] not in used:
            used.add(pick[0])
            named[label] = [pick[1][0], pick[1][1]]
    rest = [c for i, c in enumerate(dirs) if i not in used]
    if rest:
        top = min(rest, key=lambda c: c[1])
        bottom = max(rest, key=lambda c: c[1])
        if abs(top[1] - bottom[1]) > 8:
            named["U"] = [top[0], top[1]]
            named["D"] = [bottom[0], bottom[1]]
    for i, (x, y, _w) in enumerate(btns, start=1):
        named["c%d" % i] = [x, y]

    data = {"video": Path(args.video).name, "panel": [px, py],
            "panel_size": [PANEL_W, PANEL_H], "detected": len(cells),
            "cells": named}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print("\nwrote %s (%d direction + %d button cells)" % (out, len(dirs), len(btns)))
    print("check: the button count should equal the keycaps visible in the panel view.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
