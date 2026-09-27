"""Does the collected data actually contain the big jumps?

Making the double jump expressible is not the same as having it in the data.
Uniform sampling hits `jump, noop, jump` about once in 400 actions, and a jump
that never happens cannot be learned from: the IDM would keep answering with the
short hop it has only ever seen, and pi could never imitate the videos, where the
speedrunner jumps at full height and double jumps constantly.

So the data has to be measured, not assumed.  This reads a self-play session back
and, for every action step, measures how far the character actually rose in the
following steps.  The reference numbers come from the engine measurements:

    one 250 ms hold        172 px at one spot, 203 px at another
    jump,noop,jump         246 px

Absolute height depends on the landing spot, so the useful reading is the SHAPE of
the distribution: a session where nothing exceeds ~200 px contains no big jumps,
whatever the reasons.

    uv run python scripts/measure_jump_height.py runs/selfplay/selfplay-010-motif.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.actions.space import ActionSpace, buttons_from_mask  # noqa: E402

BUCKETS = (0, 50, 100, 150, 200, 250, 400)
LOOKAHEAD = 6          # steps; a jump lasts about four at 0.25 s each


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    space = ActionSpace.minimal()
    names = [", ".join(buttons_from_mask(m)) or "noop" for m in space.masks]

    for path in args.paths:
        z = np.load(path, allow_pickle=True)
        acts = z["acts"].astype(np.int64)
        states = [json.loads(s) for s in z["states"]]
        py = np.asarray([(s.get("physics") or {}).get("py", np.nan) for s in states], dtype=float)
        n = len(acts)
        jumps = [t for t in range(n) if "jump" in names[int(acts[t])]]
        rises = []
        for t in jumps:
            window = py[t : t + LOOKAHEAD + 1]
            if len(window) < 2 or not np.isfinite(window).any():
                continue
            rises.append(float(window[0] - np.nanmin(window)))
        rises = np.asarray(rises)
        print("\n=== %s ===" % Path(path).name)
        print("转移 %d | 含 jump 的步 %d（%.1f%%）" % (n, len(jumps), 100 * len(jumps) / max(1, n)))
        if not len(rises):
            print("  没有跳跃步")
            continue
        print("  跳后上升：中位 %.1f px | p90 %.1f | 最大 %.1f" %
              (np.median(rises), np.percentile(rises, 90), rises.max()))
        print("  分布：" + " | ".join(
            "%d-%d: %d" % (BUCKETS[i], BUCKETS[i + 1], int(((rises >= BUCKETS[i]) & (rises < BUCKETS[i + 1])).sum()))
            for i in range(len(BUCKETS) - 1)))
        # The double jump is the only thing beyond a single hold's ceiling, so
        # how many steps clear it is the number that matters.
        print("  超过 220 px（= 只有二段跳/长按才够得到）：%d 次（%.1f%% of jumps）"
              % (int((rises > 220).sum()), 100 * (rises > 220).mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
