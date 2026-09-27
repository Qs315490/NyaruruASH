"""What is a self-play batch actually made of?

Two different questions get conflated when data quality is judged by "did
anything change", and the confusion has cost this project its whole history:

  own_moved    the player's own body moved.  In a small room this is true for 83%
               of random steps - and almost all of it is gravity, not input.
  effective    the displacement went the way the ACTION intended.  Indoors that
               is 19%; `right, jump` is consistent 0% of the time because the
               character is pressed against something.

A pseudo-labeller taught on the second kind learns nothing, which is why it
answered the prior.  This prints both, per action as well as overall, so a
collection site can be compared before hours are spent collecting there.

    uv run python scripts/measure_selfplay.py runs/selfplay/selfplay-003-outdoor.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.actions.space import ActionSpace, buttons_from_mask  # noqa: E402
from ash.data.effect import MIN_EFFECTIVE_PX, action_intent, is_effective  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    space = ActionSpace.minimal()
    masks = space.masks
    names = [", ".join(buttons_from_mask(m)) or "noop" for m in masks]

    for path in args.paths:
        z = np.load(path, allow_pickle=True)
        acts = z["acts"].astype(np.int64)
        states = [json.loads(s) for s in z["states"]]
        n = len(acts)
        gate = [is_effective(states[t], states[t + 1], masks[int(acts[t])]) for t in range(n)]
        eff = np.array([g["effective"] is True for g in gate])
        bad = np.array([g["effective"] is False for g in gate])
        und = np.array([g["effective"] is None for g in gate])
        own = z["moved"].astype(bool)
        print("\n=== %s ===" % Path(path).name)
        print("转移 %d | 事件 %s" % (n, np.sum([action_intent(m) == (0, 0) for m in masks])))
        print("  有效（意图一致，>=%.0fpx）  %5.1f%%" % (MIN_EFFECTIVE_PX, 100 * eff.mean()))
        print("  无效（顶墙/被抵消/重力）    %5.1f%%" % (100 * bad.mean()))
        print("  无法判定（无位移意图）      %5.1f%%" % (100 * und.mean()))
        print("  对照 own_moved（旧判据）    %5.1f%%  ← 把重力也算成「在动」" % (100 * own.mean()))
        print("  逐类：")
        for cls in range(len(masks)):
            s = acts == cls
            if s.sum() < 15:
                continue
            intent = action_intent(masks[cls])
            print("    %-20s n=%4d | 有效 %5.1f%% | own_moved %5.1f%% | 意图 %s"
                  % (names[cls], s.sum(), 100 * eff[s].mean(), 100 * own[s].mean(), intent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
