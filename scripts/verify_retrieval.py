"""Live retrieval check: real game frames vs the internet corpus.

Runs the ASH retrieval path (paper Algorithm 3) end to end against real data:

  1. capture N frames from the running game over CDP;
  2. embed them with DINOv2 (the same frozen embedder the loop uses);
  3. embed the corpus videos in data/corpus;
  4. score every corpus video with greedy one-to-one window matching;
  5. report the ranking and a control: a trajectory of *random noise* frames
     must not outrank the real game frames against the same corpus.

Step 5 is the point of the script.  Retrieval that returns a plausible-looking
ordering while ignoring its inputs is the failure mode this project has hit
before (see docs/pitfalls.md): a metric that never moves looks like a metric
that works.  The noise control fails loudly if similarity is not actually
driven by the frames.

Usage:
    python scripts/verify_retrieval.py --frames 8 --corpus data/corpus
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ash.memory.embeddings import FrameEmbedder  # noqa: E402
from ash.retrieval.matching import best_window_score  # noqa: E402

CDP_HTTP = "http://127.0.0.1:9222"


async def capture_frames(count: int, stride: int, image_size: int) -> np.ndarray:
    """Pump the game and screenshot between frames; returns (count, H, W, C)."""
    import websockets

    pages = json.loads(urllib.request.urlopen(CDP_HTTP + "/json").read())
    page = next(p for p in pages if p.get("type") == "page" and p.get("webSocketDebuggerUrl"))
    ws_url = page["webSocketDebuggerUrl"]

    from ash.memory.js_source import js_source

    seq = 0

    async def call(ws, method, params=None):
        nonlocal seq
        seq += 1
        await ws.send(json.dumps({"id": seq, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") == seq:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    async def evl(ws, expr):
        r = await call(ws, "Runtime.evaluate",
                       {"expression": expr, "returnByValue": True, "awaitPromise": True})
        if r.get("exceptionDetails"):
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", "?"))
        return r.get("result", {}).get("value")

    frames: list[np.ndarray] = []
    import base64

    import cv2

    async with websockets.connect(ws_url, max_size=200 * 1024 * 1024) as ws:
        await call(ws, "Runtime.enable")
        await call(ws, "Page.enable")
        await call(ws, "Page.addScriptToEvaluateOnNewDocument", {"source": js_source()})
        await evl(ws, js_source())
        # The pump only works once the game has a ticker; wait for it.
        await evl(ws, "__ash.pump.install()")
        for i in range(count):
            await evl(ws, "__ash.pump.pump(%d)" % stride)
            shot = await call(ws, "Page.captureScreenshot", {"format": "jpeg", "quality": 90})
            raw = base64.b64decode(shot["data"])
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames).astype(np.uint8)


def load_corpus(corpus_dir: Path) -> list[tuple[str, np.ndarray]]:
    out = []
    for npz in sorted(corpus_dir.glob("*.npz")):
        out.append((npz.stem, np.load(npz)["frames"]))
    return out


def score_against(query: np.ndarray, corpus: list[tuple[str, np.ndarray]],
                  embedder: FrameEmbedder, w_r: int) -> list[tuple[str, float]]:
    q = embedder.embed(query)
    scored = []
    for vid, frames in corpus:
        e = embedder.embed(frames)
        scored.append((vid, best_window_score(q @ e.T, w_r)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--w-r", type=int, default=8)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    corpus = load_corpus(Path(args.corpus))
    if not corpus:
        print("no corpus npz under %s; run scripts/build_corpus.py first" % args.corpus,
              file=sys.stderr)
        return 2
    print("corpus: %s" % [(vid, f.shape) for vid, f in corpus])

    print("capturing %d game frames over CDP ..." % args.frames)
    game = asyncio.run(capture_frames(args.frames, args.stride, args.image_size))
    print("captured %s, %d distinct" % (game.shape, len(np.unique(game, axis=0))))

    embedder = FrameEmbedder(args.device)

    real = score_against(game, corpus, embedder, args.w_r)
    print("\nreal game frames vs corpus:")
    for vid, s in real:
        print("  %-16s %8.3f" % (vid, s))

    # Control: unrelated noise must not beat real gameplay.
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=game.shape, dtype=np.uint8)
    noise_scores = score_against(noise, corpus, embedder, args.w_r)
    print("\nnoise control vs corpus:")
    for vid, s in noise_scores:
        print("  %-16s %8.3f" % (vid, s))

    best_real = real[0][1] if real else float("-inf")
    best_noise = noise_scores[0][1] if noise_scores else float("-inf")
    print("\nbest real %.3f vs best noise %.3f" % (best_real, best_noise))
    ok = best_real > best_noise
    print("VERDICT:", "PASS - similarity tracks content" if ok
          else "FAIL - similarity ignores content")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
