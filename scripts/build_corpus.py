"""Build the internet-video corpus D^I from speedrun recordings.

ASH learns from observation-only video, so the corpus is a set of frame
sequences with no action labels.  This script turns URLs into the npz format the
retrieval code expects:

    data/corpus/<video_id>.npz   key "frames": (T, H, W, C) uint8 RGB

Pipeline per video: yt-dlp download -> sample at --fps -> resize to
--image-size -> npz.  Frames are sampled uniformly so a 40-minute speedrun
becomes a few thousand frames rather than 140k, which is what the paper's
"2 second interval" indexing approximates.

Two deliberate choices:

- Downloads are capped at 480p.  The corpus is consumed at 256x256, so a 1080p
  source only buys decode time (a 1.1 GB file decodes for minutes and yields the
  same 256x256 frames as its 480p encode).
- Frames are extracted through ffmpeg's fps filter rather than a per-frame
  OpenCV read loop.  cv2.VideoCapture().read() on a long HLS-merged mp4 is slow
  and can silently return repeated frames; ffmpeg decodes once, filters, and
  streams rawvideo, which is both faster and verifiable.

Usage:
    python scripts/build_corpus.py --url URL [--url URL ...]
    python scripts/build_corpus.py --search "nyaruru speedrun" --limit 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ash.config import load_game_config  # noqa: E402

# Resolve yt-dlp through the running interpreter: a bare "yt-dlp" is not on PATH
# when the venv is not activated, which is the normal case for `uv run` and for
# CI.  `python -m yt_dlp` always resolves to the installed package.
YTDLP = [sys.executable, "-m", "yt_dlp"]

# 480p is the ceiling for corpus downloads; see the module docstring.
MAX_HEIGHT = 480


def search_videos(query: str, limit: int) -> list[tuple[str, str]]:
    """Return (video_id, title) via yt-dlp's search extractor."""
    proc = subprocess.run(
        [*YTDLP, "--flat-playlist", "--print", "%(id)s|%(title)s",
         "ytsearch%d:%s" % (limit, query)],
        capture_output=True, text=True, check=False,
    )
    out = []
    for line in proc.stdout.splitlines():
        if "|" in line:
            vid, title = line.split("|", 1)
            out.append((vid.strip(), title.strip()))
    return out


def download(url: str, out_dir: Path) -> Path:
    """Download an mp4 capped at MAX_HEIGHT; returns the local path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("source.mp4"))
    if existing and existing[0].stat().st_size > 0:
        return existing[0]
    tmpl = str(out_dir / "source.%(ext)s")
    fmt = (
        "bv*[height<=%d][ext=mp4]+ba[ext=m4a]/b[height<=%d][ext=mp4]/b[height<=%d]/b"
        % (MAX_HEIGHT, MAX_HEIGHT, MAX_HEIGHT)
    )
    subprocess.run(
        [*YTDLP, "-f", fmt, "--merge-output-format", "mp4", "--no-playlist",
         "-o", tmpl, url],
        check=True,
    )
    cands = sorted(out_dir.glob("source.*"))
    cands = [c for c in cands if c.suffix in (".mp4", ".mkv", ".webm")]
    if not cands:
        raise RuntimeError("yt-dlp produced no file for %s" % url)
    return cands[0]


def extract_frames(video: Path, fps: float, image_size: int) -> np.ndarray:
    """Sample `fps` frames per second and resize to (image_size, image_size).

    One ffmpeg pass: decode, resample with the fps filter, scale, emit rgb24
    rawvideo.  Reshaping the byte stream gives the (T, H, W, C) array directly.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(video),
        "-vf", "fps=%g,scale=%d:%d:flags=area" % (fps, image_size, image_size),
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed on %s: %s" % (video, proc.stderr.decode("utf-8", "replace")[:400])
        )
    frame_bytes = image_size * image_size * 3
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(buf) // frame_bytes
    if n == 0:
        raise RuntimeError("no frames decoded from %s" % video)
    return buf[: n * frame_bytes].reshape(n, image_size, image_size, 3)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", action="append", default=[])
    p.add_argument("--search", default=None, help="yt-dlp search query")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--out", default="data/corpus")
    p.add_argument("--work", default="data/corpus/.downloads")
    # Default to the game's control rate, NOT an arbitrary number: the IDM is
    # trained on the agent's timestep and applied to these frames, so a corpus
    # sampled at another rate makes the IDM label 2 s changes with a model that
    # only ever saw 0.25 s ones.  That mismatch collapsed the pseudo-actions to
    # a single class and looked like the policy "converging".
    p.add_argument("--fps", type=float, default=None,
                   help="sampled frames/second; default = 1/control_interval_s "
                        "from config/game.yaml (4 fps = 0.25 s)")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--video", default=None,
                   help="re-extract from an already downloaded file (skip download)")
    args = p.parse_args(argv)
    if args.fps is None:
        args.fps = load_game_config().corpus_fps

    targets: list[tuple[str, str]] = []
    if args.search:
        # search_videos yields (id, title); the downloader needs a URL, so the
        # canonical watch URL is rebuilt from the id.
        for vid, title in search_videos(args.search, args.limit):
            targets.append((vid, "https://www.youtube.com/watch?v=%s" % vid))
    for url in args.url:
        vid = url.rstrip("/").split("=")[-1].split("/")[-1]
        targets.append((vid, url))
    if args.video:
        vpath = Path(args.video)
        targets = [(vpath.parent.name or vpath.stem, str(vpath))]
    if not targets:
        print("no targets; pass --url or --search", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    work = Path(args.work)
    # Merge into the existing manifest instead of replacing it: the corpus is
    # grown video by video, and a manifest that only lists the most recent
    # invocation silently loses how the rest of D^I was built.
    manifest_path = out_dir / "manifest.json"
    entries: dict[str, dict] = {}
    if manifest_path.exists():
        for old in json.loads(manifest_path.read_text()):
            entries[old["id"]] = old
    for vid, ref in targets:
        try:
            if args.video:
                video = Path(ref)
            else:
                video = download(ref, work / vid)
            frames = extract_frames(video, args.fps, args.image_size)
        except Exception as e:
            print("skip %s: %s" % (vid, e), file=sys.stderr)
            continue
        # Sanity check: a decode bug that returns one frame repeatedly would
        # otherwise produce a corpus that looks fine and teaches nothing.
        uniq = len(np.unique(frames.reshape(len(frames), -1), axis=0))
        np.savez_compressed(out_dir / ("%s.npz" % vid), frames=frames)
        entries[vid] = {
            "id": vid, "title": ref, "frames": int(len(frames)),
            "image_size": args.image_size, "fps": args.fps,
            "distinct_frames": int(uniq),
        }
        print("ok %s: %d frames (%d distinct)" % (vid, len(frames), uniq))

    manifest = list(entries.values())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("wrote %d videos to %s" % (len(manifest), out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
