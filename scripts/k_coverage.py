"""Why are there no key moments where the agent actually is?

K keeps a cluster only if it spans >= `min_distinct_trajectories` trajectories
(kdm.py).  Fitting K on the scraped speedruns alone therefore cannot produce a key
moment in a room that only one video ever visits - and the agent's rooms are
exactly that.  This measures the situation on the real corpus and on a recorded
session, so the remedy is chosen from numbers rather than from the filter's
description.

    uv run python scripts/k_coverage.py --recording data/recordings/human-001.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.memory.embeddings import FrameEmbedder  # noqa: E402
from ash.memory.kdm import KeyMomentModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/corpus-embeddings.npz")
    ap.add_argument("--recording", action="append", default=[],
                    help="recorded session; repeatable, each is its own trajectory")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with np.load(args.index) as data:
        videos = [(k, data[k]) for k in data.files if k != "__meta__"]
    print("corpus trajectories: %d (%d frames)"
          % (len(videos), sum(v.shape[0] for _, v in videos)))

    embs, ids = [], []
    recordings: list[str] = []
    embedder = None
    for name, e in videos:
        embs.append(e)
        ids.extend([name] * len(e))
    for path in args.recording:
        from ash.train.demo_mapping import load_demo

        if embedder is None:
            embedder = FrameEmbedder(device=args.device, image_size=args.image_size)
        frames = np.asarray(load_demo(path)["observations"])
        rec = embedder.embed(frames)
        stem = Path(path).stem
        nearest = max(float((rec @ e.T).max()) for _, e in videos)
        print("recording %-12s %5d frames | nearest corpus cosine %.4f"
              % (stem, len(rec), nearest))
        embs.append(rec)
        ids.extend([stem] * len(rec))
        recordings.append(stem)
    if args.recording:
        print()

    matrix = np.concatenate(embs).astype(np.float32)
    ids = np.asarray(ids)
    print()

    # The threshold only decides which clusters survive, not how the clustering
    # itself comes out, so one fit answers all three questions.
    kdm = KeyMomentModel(min_distinct_trajectories=3)
    report = kdm.fit(matrix, ids)
    labels = kdm._labels          # fit-time labelling; cluster_of() is for new points
    spread = {}
    for label in np.unique(labels):
        if label < 0:
            continue
        members = ids[labels == label]
        spread[int(label)] = (len(np.unique(members)), len(members))

    print("clusters total %d, noise %.1f%%"
          % (report["clusters_total"], 100.0 * report["noise_rate"]))
    histogram: dict[int, int] = {}
    for trajectories, _ in spread.values():
        bucket = min(trajectories, 4)
        histogram[bucket] = histogram.get(bucket, 0) + 1
    print("clusters by how many trajectories they span:",
          {("%d" % k if k < 4 else "4+"): v for k, v in sorted(histogram.items())})
    for threshold in (3, 2, 1):
        kept = [cid for cid, (t, _) in spread.items() if t >= threshold]
        frames = sum(spread[cid][1] for cid in kept)
        rec_only = sum(1 for cid in kept if threshold == 1 and
                       set(ids[labels == cid]) == {"recording"}) if args.recording else 0
        print("  threshold %d -> %d clusters kept (%.1f%% of clustered frames)%s"
              % (threshold, len(kept), 100.0 * frames / max(1, sum(v[1] for v in spread.values())),
                 "; of which recording-only: %d" % rec_only if threshold == 1 else ""))
    if recordings:
        print()
        for stem in recordings:
            mask = ids == stem
            covered = np.isin(labels[mask],
                              [c for c, (t, _) in spread.items() if t >= 3])
            print("  %-12s frames in a kept (>=3-trajectory) cluster: %5d / %5d (%.1f%%)"
                  % (stem, covered.sum(), mask.sum(), 100.0 * covered.mean()))
        # The point of recording the same room three times: does it now span
        # three trajectories and therefore survive the filter?
        shared = [cid for cid, _ in spread.items()
                  if len({i for i in set(ids[labels == cid]) if i in set(recordings)}) >= 2]
        for cid in sorted(shared):
            members = [i for i in dict.fromkeys(ids[labels == cid])]
            print("  cluster %d: %d frames across trajectories %s" % (cid, spread[cid][1], members))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
