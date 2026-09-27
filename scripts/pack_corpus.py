"""Rewrite the corpus npz files as uncompressed, memory-mappable .npy.

The retrieval corpus is stored deflated.  `np.load(npz)["frames"]` materializes a
video at its capture resolution - 2 to 2.6 GB each - as *anonymous* memory the
kernel cannot reclaim, and on this machine that pushed the system into swap: the
per-video bootstrap phase stalled for minutes on a 6-second decompression, which
is the stall behind three separate rounds.

An uncompressed .npy is read with `mmap_mode="r"`, so the frames live in the page
cache as clean, evictable pages instead of pinned process memory.  The bytes are
identical - only the container changes - so cached DINOv2 embeddings stay valid.

    uv run python scripts/pack_corpus.py --remove-source
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default="data/corpus")
    ap.add_argument("--remove-source", action="store_true",
                    help="delete each npz once its .npy is written and verified")
    args = ap.parse_args()
    base = Path(args.corpus)
    npzs = sorted(base.glob("*.npz"))
    print("packing %d videos in %s" % (len(npzs), base))
    for npz in npzs:
        target = npz.with_suffix(".npy")
        if target.exists():
            print("  %-16s already packed (%.2f GB)" % (npz.stem, target.stat().st_size / 1e9))
            if args.remove_source:
                npz.unlink()
            continue
        with np.load(npz) as data:
            frames = data["frames"]
            np.save(target, frames)
        # Verify the round trip before considering the source expendable: a
        # truncated .npy would silently become the corpus.
        check = np.load(target, mmap_mode="r")
        ok = check.shape == frames.shape and check.dtype == frames.dtype
        del frames, check
        print("  %-16s %s -> %.2f GB%s" % (npz.stem, "ok" if ok else "MISMATCH",
                                           target.stat().st_size / 1e9,
                                           " (source removed)" if args.remove_source else ""))
        if args.remove_source:
            if not ok:
                raise SystemExit("refusing to remove %s: round trip failed" % npz)
            npz.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
