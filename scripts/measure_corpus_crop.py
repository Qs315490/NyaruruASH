"""Does cropping the speedrun frames make them usable as K anchors?

The scraped videos carry LiveSplit panels, a banner and a scaled/offset game
view, so their frames sit ~0.85-0.89 cosine from a live capture while a recording
made through the same CDP path sits at ~0.96.  Cropping to the game region is the
obvious remedy; whether it is ENOUGH is a measurement, not an argument - K judges
in its own PCA/cluster space, not by raw cosine.

    uv run python scripts/measure_corpus_crop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.memory.embeddings import FrameEmbedder  # noqa: E402
from ash.memory.kdm import KeyMomentModel  # noqa: E402

#: Rects found by searching for the crop that maximises cosine against a live
#: frame (x0, y0, x1, y1 as fractions).  Per video: the layout differs.
CROPS = {
    "0r2lVc1uKa0": (0.00, 0.00, 0.85, 0.85),
    "61no6YQJLuQ": (0.00, 0.00, 1.00, 0.80),
    "BV13HnzzPEEN": (0.20, 0.15, 1.00, 1.00),
    "BV19s4y1y7un": (0.25, 0.05, 1.00, 0.80),
    "BV1Am4y1t7Sv": (0.20, 0.05, 0.85, 0.80),
    "eIuH528wp4c": (0.00, 0.00, 1.00, 1.00),
}


def main() -> int:
    emb = FrameEmbedder(device="cuda", image_size=256)
    live = cv2.cvtColor(cv2.imread("runs/live-map6.png"), cv2.COLOR_BGR2RGB)
    live_emb = emb.embed(live[None])[0]

    out: dict[str, np.ndarray] = {}
    for stem, (x0, y0, x1, y1) in CROPS.items():
        arr = np.load("data/corpus/%s.npy" % stem, mmap_mode="r")
        # (T, H, W, C): the spatial axes are 1 and 2.  Slicing the first two
        # indexes crops TIME, which is what an earlier version of this script did
        # - it embedded 217 "frames" of a 9405-frame video and reported a
        # meaningless similarity.
        n, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
        ys, xs = slice(int(h * y0), int(h * y1)), slice(int(w * x0), int(w * x1))
        chunks = []
        for i in range(0, n, 256):
            block = np.asarray(arr[i:i + 256, ys, xs])
            chunks.append(emb.embed(np.stack([
                cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA) for f in block])))
        out[stem] = np.concatenate(chunks)
        print("%-14s %d frames -> nearest live cosine %.4f"
              % (stem, len(out[stem]), float((out[stem] @ live_emb).max())))
    np.savez_compressed("data/corpus-embeddings-cropped.npz", **out)

    # The verdict that matters: fit K exactly as a run would and ask whether the
    # live frame lands in a KEPT cluster (its own PCA/cluster space, not cosine).
    index = np.load("data/corpus-embeddings.npz")
    recs = ["house-001", "house-002", "house-003", "human-001"]
    embs, ids = [], []
    for k, e in out.items():
        embs.append(e); ids.extend([k] * len(e))
    for r in recs:
        e = np.load("data/recordings/%s.npy" % r, mmap_mode="r") if Path(
            "data/recordings/%s.npy" % r).exists() else None
        if e is None:
            from ash.train.demo_mapping import load_demo
            frames = np.asarray(load_demo("data/recordings/%s.npz" % r)["observations"])
            e = emb.embed(frames)
        embs.append(e); ids.extend([r] * len(e))
    del index
    kdm = KeyMomentModel()
    report = kdm.fit(np.concatenate(embs).astype(np.float32), ids)
    print("K on cropped corpus + recordings:", report)
    flags = kdm.classify_sequence(live_emb[None])
    print("live frame in a kept (>=3-trajectory) cluster:", bool(flags[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
