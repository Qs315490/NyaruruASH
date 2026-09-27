"""Convert the demonstration npz into the memory-mappable form.

`data/idm-human.npz` is deflated, so reading it materializes all 24447 frames at
128x128x3 - 1.2 GB - and a run that also holds corpus videos can be OOM-killed by
the kernel for it.  Uncompressed `.npy` files can be memory-mapped, so a round
only faults in the windows it actually samples.

    uv run python scripts/pack_demos.py data/idm-human.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.train.demo_mapping import pack_demo  # noqa: E402


def main(argv: list[str]) -> int:
    src = Path(argv[1]) if len(argv) > 1 else Path("data/idm-human.npz")
    dst = pack_demo(src)
    print("packed %s -> %s" % (src, dst))
    for child in sorted(dst.glob("*.npy")):
        print("  %-20s %.1f MB" % (child.name, child.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
