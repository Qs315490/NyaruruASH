"""Verify the left/right SEMANTICS of a video's labels against picture motion.

Why this exists: the direction cells come from geometry (clustering the ball's nine positions,
or the positions of four arrow keycaps), and which one is "left" is an assumption - nothing in
the pixels says so.  When the labels are wrong the model is blamed instead: in fold B the
predictions called `right` "left" at every opportunity while `right` had the most positives of
any key in that video, which is exactly what swapped labels would look like.

The independent signal is the camera: it follows the player, so a player running right moves
the BACKGROUND left.  So for ticks where the label says `right`, the frame-to-frame background
shift must be negative, and for `left` it must be positive.  Verification is the two signs
coming out opposite; agreement with the sign convention is what makes the labels usable.

    uv run python scripts/verify_direction_labels.py --video data/video-src/<stem>.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.data.video_pack import artifact, stem_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps", type=int, default=4,
                    help="sampling rate; at corpus_fps (4) frame i is tick i, 1:1")
    ap.add_argument("--region", required=True,
                    help="x0,y0,x1,y1 in NATIVE pixels: a TEXTURED part of the game area. "
                         "A smooth sky correlates to ~0 no matter what the player does, which "
                         "is what made video 2 look motionless on the first attempt.")
    ap.add_argument("--tick", type=float, default=0.25)
    args = ap.parse_args()
    video = Path(args.video)
    stem = stem_of(video)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True).stdout.split()
    vw, vh = int(probe[0]), int(probe[1])
    W, H = 640, 360
    x0, y0, x1, y1 = (int(v) for v in args.region.split(","))
    sx, sy = W / vw, H / vh
    gx0, gy0, gx1, gy1 = int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)
    if gx1 - gx0 < 32 or gy1 - gy0 < 32:
        print("region too small after scaling: %d x %d" % (gx1 - gx0, gy1 - gy0))
        return 2
    # Streamed: one frame at a time, no 3 GB buffer for an hour of 4 fps frames.
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", "fps=%d,scale=%d:%d" % (args.fps, W, H),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    dx, n, prev = [], 0, None
    while True:
        buf = proc.stdout.read(W * H)
        if not buf or len(buf) < W * H:
            break
        g = np.frombuffer(buf, np.uint8).reshape(H, W)[gy0:gy1, gx0:gx1].astype(np.float32)
        dx.append(0.0 if prev is None else cv2.phaseCorrelate(prev, g)[0][0])
        prev = g
        n += 1
    proc.wait()
    dx = np.asarray(dx)
    print("%s %dx%d: %d frames at %dfps, region %s -> %dx%d"
          % (stem, vw, vh, n, args.fps, args.region, gx1 - gx0, gy1 - gy0))

    L = np.load(artifact(stem, "labels"), allow_pickle=True)
    held, names = L["held"], [str(x) for x in L["names"]]
    idx = {k: i for i, k in enumerate(names)}
    # frame i (at args.fps) sits at tick round(i / fps / tick)
    tick = np.round(np.arange(n) / args.fps / args.tick).astype(int)
    tick = np.clip(tick, 0, len(held) - 1)

    print("%-8s %6s %14s  %s" % ("label", "frames", "mean dx", "expected"))
    res = {}
    for key, want in (("right", "negative (camera follows)"), ("left", "positive")):
        if key not in idx:
            print("%-8s (not in this video's labels)" % key)
            continue
        v = held[tick, idx[key]].copy(); v[0] = False
        m = float(dx[v].mean()) if v.sum() else float("nan")
        res[key] = (int(v.sum()), m)
        print("%-8s %6d %+14.2f  %s" % (key, int(v.sum()), m, want))
    noop = ~held[tick].any(axis=1); noop[0] = False
    print("%-8s %6d %+14.2f  near zero" % ("noop", int(noop.sum()),
                                           float(dx[noop].mean()) if noop.sum() else float("nan")))
    if "right" in res and "left" in res:
        # Read the DIFFERENCE: a common drift (auto-scroll, camera shake) cancels, and only
        # the part of the motion that distinguishes left from right survives.
        diff = res["right"][1] - res["left"][1]
        ok = diff < -0.3 and res["right"][1] < res["left"][1] - 0.3
        print("\nright - left = %+.2f px  (must be clearly NEGATIVE: right runs one way, "
              "left the other)" % diff)
        print("VERDICT: %s" % ("labels agree with picture motion" if ok else
              "*** cannot confirm: either swapped, or this region/scene has no usable motion ***"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
