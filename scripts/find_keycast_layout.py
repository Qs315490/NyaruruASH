"""Find a video's keycast layout from the pixels: panel origin + every keycap.

The three speedrun videos each arrange KeyDisplay differently - the positions are
configured per session - so a layout cannot be inherited.  Hand-reading a screenshot
worked once and missed two keycaps; this does it from data:

  * sample frames across the whole video;
  * for each pixel take the 99th percentile MINUS the median over time.  A keycap
    lights up sometimes, so its pixels jump; static overlay furniture does not move
    at all, and the panel background stays flat.  This separates "sometimes bright"
    from both "always bright" and "never bright", which a simple max or std does not;
  * connected components of that map give the keycap-sized blobs (including the
    rarely used ones, which is the whole point);
  * the left group of nine positions arranged as a 3x3 is the direction grid.

Output is the same JSON the page and the extractor already read, so nothing else
needs to change: panel = bounding box of the cells plus a margin, cells relative to it.

    uv run python scripts/find_keycast_layout.py --video data/video-src/BV1Am4y1t7Sv.mp4
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

DIR_NAMES = ["U", "UL", "UR", "L", "-", "R", "DL", "DR", "D"]


def sample(video: Path, n: int, scale: int) -> np.ndarray:
    """Frames as (n, H, W) gray at 1/scale of the native size."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True)
    w, h, dur = (float(v) for v in probe.stdout.split())
    w, h = int(w) // scale, int(h) // scale
    out = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "%.2f" % t, "-i", str(video),
             "-frames:v", "1", "-vf", "scale=%d:%d" % (w, h),
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=True).stdout
        if len(raw) == w * h:
            out.append(np.frombuffer(raw, np.uint8).reshape(h, w))
    return np.stack(out) if out else np.zeros((0, h, w), np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--swing", type=float, default=60.0,
                    help="how much brighter than its median a pixel must get")
    ap.add_argument("--margin", type=int, default=26, help="panel margin around the cells")
    ap.add_argument("--area-min", type=int, default=40, help="keycap blob area, samples scaled")
    ap.add_argument("--area-max", type=int, default=4000,
                    help="keycap blob area ceiling - a 1080p keycap is ~4x a 720p one")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    video = Path(args.video)
    A = sample(video, args.samples, args.scale)
    if len(A) < 10:
        print("could not sample frames")
        return 2
    print("sampled %d frames at %dx%d (1/%d scale)"
          % (len(A), A.shape[2], A.shape[1], args.scale))
    med = np.median(A, axis=0)
    hi = np.percentile(A, 99, axis=0)
    swing = hi - med                      # "sometimes bright"
    m = (swing > args.swing).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    blobs = []
    for j in range(1, n):
        a = stats[j, cv2.CC_STAT_AREA]
        bw, bh = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
        if args.area_min <= a <= args.area_max and 0.5 <= bw / max(1, bh) <= 2.0:
            blobs.append((float(cent[j][0]), float(cent[j][1]), int(a)))
    print("%d keycap-sized 'sometimes bright' blobs (area %d..%d)"
          % (len(blobs), args.area_min, args.area_max))
    if blobs:
        areas = np.array([b[2] for b in blobs])
        print("   blob area quartiles %s" % np.percentile(areas, [25, 50, 75]).round(0))
    # cluster to 12 px (half-scale), greedy
    clusters: list[dict] = []
    for x, y, a in blobs:
        for c in clusters:
            if np.hypot(c["x"] / c["n"] - x, c["y"] / c["n"] - y) < 12:
                c["x"] += x; c["y"] += y; c["n"] += 1
                break
        else:
            clusters.append({"x": x, "y": y, "n": 1})
    clusters = [c for c in clusters if c["n"] >= 2]
    pts = sorted([(c["x"] / c["n"], c["y"] / c["n"], c["n"]) for c in clusters],
                 key=lambda p: (round(p[1] / 8), p[0]))
    print("%d stable positions:" % len(pts))
    for x, y, k in pts:
        print("   x=%6.1f y=%6.1f  hits=%d" % (x * args.scale, y * args.scale, k))
    if not pts:
        print("nothing found - check --swing (the panel may be dimmer here)")
        return 2

    # the direction grid: the nine leftmost points forming a 3x3
    xs = np.array([p[0] for p in pts])
    left_cut = np.percentile(xs, 35)
    left = [p for p in pts if p[0] <= left_cut]
    dirs: dict[str, list[float]] = {}
    if len(left) >= 5:
        cx = np.mean([p[0] for p in left]); cy = np.mean([p[1] for p in left])
        rel = sorted(left, key=lambda p: (np.arctan2(p[1] - cy, p[0] - cx)))
        # nearest name by angle among the eight spokes, or centre
        for x, y, _k in left:
            dx, dy = x - cx, y - cy
            if np.hypot(dx, dy) < 6:
                dirs["-"] = [x * args.scale, y * args.scale]
                continue
            ang = np.degrees(np.arctan2(dy, dx))
            spokes = {"R": 0, "DR": 45, "D": 90, "DL": 135, "L": 180,
                      "UL": -135, "U": -90, "UR": -45}
            best = min(spokes, key=lambda s: abs((ang - spokes[s] + 180) % 360 - 180))
            if best not in dirs:
                dirs[best] = [x * args.scale, y * args.scale]
    for k in ("L", "R", "U", "D"):
        if k in dirs:
            continue
        # fall back to the extreme point of the left group along that axis
        if k == "L":
            p = min(left, key=lambda q: q[0])
        elif k == "R":
            p = max(left, key=lambda q: q[0])
        elif k == "U":
            p = min(left, key=lambda q: q[1])
        else:
            p = max(left, key=lambda q: q[1])
        dirs[k] = [p[0] * args.scale, p[1] * args.scale]

    right = [p for p in pts if p[0] > left_cut]
    cells = {k: [round(v[0]), round(v[1])] for k, v in dirs.items()}
    for i, (x, y, _k) in enumerate(right, start=1):
        cells["c%d" % i] = [round(x * args.scale), round(y * args.scale)]
    all_pts = np.array([[c[0], c[1]] for c in cells.values()])
    x0 = max(0, int(all_pts[:, 0].min()) - args.margin)
    y0 = max(0, int(all_pts[:, 1].min()) - args.margin)
    x1 = int(all_pts[:, 0].max()) + args.margin
    y1 = int(all_pts[:, 1].max()) + args.margin
    rel = {k: [int(v[0]) - x0, int(v[1]) - y0] for k, v in cells.items()}
    proto = {k: [v[0], v[1]] for k, v in rel.items() if k in DIR_NAMES}
    out = Path(args.out) if args.out else \
        Path("runs") / ("keycast-cells-%s.json" % video.stem)
    out.write_text(json.dumps(
        {"video": video.name, "panel": [x0, y0], "panel_size": [x1 - x0, y1 - y0],
         "cells": rel, "dir_proto": proto}, indent=1, ensure_ascii=False))
    print("\npanel origin (%d,%d) size %dx%d" % (x0, y0, x1 - x0, y1 - y0))
    print("cells: %d direction + %d button -> %s"
          % (len(proto), len(rel) - len(proto), out))
    print("NOTE: L/R still have to be confirmed by camera shift (they are assigned by")
    print("      geometry here, and the videos are not guaranteed to agree).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
