"""Find each video's game-content rectangle from the frames themselves.

The overlay is STATIC - a splits panel, a banner, a gamepad icon, black bars -
while the game moves, so a per-axis motion profile locates both boundaries.  The
profiles are cleanly bimodal (overlay 6-22, game 56-70 on a 0-255 scale).

Two tests that looked more natural fail, and both were caught by looking at the
result rather than at the numbers:

- per-column *saturation* (a grey panel is not a colourful game): five of the six
  videos put the panel on the LEFT and make it translucent, so the game's colours
  show through and the saturation never drops;
- per-row *brightness* (a letterbox is dark): one video's bottom banner is bright,
  and the profile picked a 37-pixel strip of banner as the game.

The layout is worth knowing anyway: five videos share one layout (panel left from
x=0.1875, banner below y=0.8125) and one mirrors it (panel right of x=0.8125).

Aspect is corrected too: the stored frames hold a square game area while the live
window is 4:3, so the embedder squashes one and not the other.

    uv run python scripts/detect_game_rect.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CORPUS = Path("data/corpus")
SAMPLES = 120
#: Skip the first and last fifth: the middle of a video can be a black transition,
#: and an earlier contact sheet sampled exactly that and showed a black tile.
WINDOW = (0.2, 0.8)


def detect(stem: str) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) in pixels, for 256x256 frames."""
    arr = np.load(CORPUS / f"{stem}.npy", mmap_mode="r")
    start, end = int(len(arr) * WINDOW[0]), int(len(arr) * WINDOW[1])
    idx = np.linspace(start, end, min(SAMPLES, end - start)).astype(int)
    std = np.asarray(arr[idx]).astype(np.float32).std(axis=0).mean(axis=2)

    def span(profile: np.ndarray) -> tuple[int, int]:
        live = profile >= 0.5 * profile.max()
        runs: list[tuple[int, int]] = []
        begin = None
        for i, value in enumerate(live):
            if value and begin is None:
                begin = i
            elif not value and begin is not None:
                runs.append((begin, i))
                begin = None
        if begin is not None:
            runs.append((begin, len(live)))
        return max(runs, key=lambda r: r[1] - r[0])

    x0, x1 = span(std.mean(axis=0))
    y0, y1 = span(std.mean(axis=1))
    return x0, y0, x1, y1


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", metavar="JSON", nargs="?", const="", default=None,
                        help="write data/corpus-crops.json for the loader to read")
    args = parser.parse_args()

    rects = {}
    for path in sorted(CORPUS.glob("*.npy")):
        rect = detect(path.stem)
        rects[path.stem] = list(rect)
        print("%-14s x[%3d,%3d] y[%3d,%3d]   frac x[%.4f,%.4f] y[%.4f,%.4f]"
              % (path.stem, rect[0], rect[2], rect[1], rect[3],
                 rect[0] / 256, rect[2] / 256, rect[1] / 256, rect[3] / 256))
    if args.write is not None:
        out = Path(args.write) if args.write else Path("data/corpus-crops.json")
        out.write_text(json.dumps({"version": 1, "rects": rects}, indent=2,
                                  sort_keys=True) + "\n")
        print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
