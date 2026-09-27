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

from ash.data.video_pack import artifact  # noqa: E402

from ash.memory.embeddings import FrameEmbedder  # noqa: E402
from ash.memory.kdm import KeyMomentModel  # noqa: E402

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_game_rect import detect  # noqa: E402


def crop_43(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Crop to the game rectangle and restore the live window's 4:3 aspect.

    The stored frames hold a square game area while the live window is 4:3, so the
    embedder squashes the live frame and not the video frame unless this is undone.
    """
    x0, y0, x1, y1 = rect
    piece = frame[y0:y1, x0:x1]
    height = piece.shape[0]
    return cv2.resize(piece, (int(height * 4 / 3), height), interpolation=cv2.INTER_AREA)


def main() -> int:
    emb = FrameEmbedder(device="cuda", image_size=256)
    live = cv2.cvtColor(cv2.imread("runs/live-map6.png"), cv2.COLOR_BGR2RGB)
    live_emb = emb.embed(live[None])[0]

    out: dict[str, np.ndarray] = {}
    for stem in sorted(p.stem for p in Path("data/corpus").glob("*.npy")):
        rect = detect(stem)
        arr = np.load(artifact(stem, "corpus-4fps"), mmap_mode="r")
        # (T, H, W, C): the spatial axes are 1 and 2.  Slicing the first two
        # indexes crops TIME, which is what an earlier version of this script did
        # - it embedded 217 "frames" of a 9405-frame video and reported a
        # meaningless similarity.
        n = arr.shape[0]
        chunks = []
        for i in range(0, n, 256):
            block = np.asarray(arr[i:i + 256])
            chunks.append(emb.embed(np.stack([
                cv2.resize(crop_43(f, rect), (256, 256), interpolation=cv2.INTER_AREA)
                for f in block])))
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
