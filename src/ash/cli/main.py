"""ASH CLI: doctor, run, eval.

The commands mirror the ASH cycle:

    ash doctor        environment self-check (CDP, game, GPU, corpus)
    ash run           inference rounds with automatic bootstrapping; K is fit
                      on the corpus at startup, then refit on D^R each round
    ash eval          print a loop report as JSON

Every command works with --backend fake for offline dry runs; the CDP backend
is only required for real play.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ash.data.corpus_crop import CroppedFrames, load_crops

log = logging.getLogger("ash")


def _setup_logging(verbose: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbose, logging.DEBUG)
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def cmd_doctor(args: argparse.Namespace) -> int:

    checks: list[tuple[str, str]] = []
    try:
        import torch

        checks.append(("torch", torch.__version__))
        checks.append(("cuda-available", str(torch.cuda.is_available())))
    except ImportError:
        checks.append(("torch", "MISSING (pip install .[train])"))
    try:
        import hdbscan  # noqa: F401

        checks.append(("hdbscan", "ok"))
    except ImportError:
        checks.append(("hdbscan", "MISSING"))
    try:
        import cv2

        checks.append(("opencv", cv2.__version__))
    except ImportError:
        checks.append(("opencv", "MISSING"))
    try:
        env = _make_env(args.backend)
        obs = env.reset()
        checks.append((f"backend:{args.backend}", f"obs {obs.rgb.shape}"))
        env.close()
    except Exception as e:  # pragma: no cover - diagnostic path
        checks.append((f"backend:{args.backend}", f"FAIL: {e}"))
    for name, status in checks:
        print(f"  {name:<24} {status}")
    return 0


def _make_env(backend: str, difficulty=None):
    """Construct a backend, capturing at the embedder's canonical frame size.

    The CDP backend defaults to a 128x128 capture, but the corpus is extracted
    at DEFAULT_IMAGE_SIZE and DINOv2 features depend on the resolution they are
    computed at - so a smaller capture would make every live frame
    systematically unlike the corpus K was fit on.  Backends that own their
    observation size (the fake one) take no resize argument.
    """
    from ash.env.base import make_env
    from ash.memory.embeddings import DEFAULT_IMAGE_SIZE

    if backend == "cdp":
        # drive="realtime": never take the engine's loop away from it.  Measured
        # with the frame pump, this game's hurt state pins at _pRealState 6 and a
        # trap that teleports correctly under the engine's own loop never
        # resolves - so a pumped self-play run collects trajectories of a
        # character that cannot move.  Determinism is not worth that here; it is
        # needed for replay/search, not for gathering behaviour.
        return make_env(
            backend,
            resize=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
            drive="realtime",
            difficulty=difficulty,
        )
    return make_env(backend)


def _check_corpus_interval(corpus_dir: str | None, expected_fps: float) -> str | None:
    """Complain when the corpus was sampled at a different rate than we act at.

    This is the failure that wasted the first live rounds: the agent stepped one
    game frame (1/60 s) while the corpus held frames 2 s apart, so the IDM was
    trained on one time scale and applied at another and collapsed to a constant
    class.  Nothing in the logs looked wrong.  Check the manifest and say so.
    """
    manifest = Path(corpus_dir or "data/corpus") / "manifest.json"
    if not manifest.exists():
        return None
    try:
        entries = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    seen = {float(e["fps"]) for e in entries if isinstance(e, dict) and "fps" in e}
    if not seen:
        return None
    if any(abs(fps - expected_fps) > 1e-6 for fps in seen):
        return (
            "corpus under %s was sampled at %s fps but the control interval "
            "(%.3f s) implies %.3f fps; the IDM is trained on the agent's "
            "timestep and applied to the corpus, so these must match "
            "(re-extract with scripts/build_corpus.py --fps %g)"
            % (corpus_dir or "data/corpus", sorted(seen),
               expected_fps and 1.0 / expected_fps, expected_fps, expected_fps)
        )
    return None


def cmd_run(args: argparse.Namespace) -> int:
    from ash.config import load_game_config
    from ash.loop.orchestrator import (
        LoopConfig,
        Orchestrator,
        load_or_init_idm,
        load_or_init_policy,
    )
    from ash.loop.runner import InferenceRunner
    from ash.utils.device import resolve_device

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    game = load_game_config()
    log.info(
        "control interval %.3f s -> %d game frames/action, corpus %.3f fps",
        game.control_interval_s, game.control_frame_skip, game.corpus_fps,
    )
    if args.backend == "cdp":
        # Say which drive mode the round will use: the difference is not
        # cosmetic (a hand-driven ticker pins this game's hurt state and the
        # trap never resolves), so it must be visible in every run's log.
        log.info("live drive mode: realtime (engine keeps its own ticker)")
    mismatch = _check_corpus_interval(args.corpus, game.corpus_fps)
    if mismatch:
        log.error("corpus/agent time scales disagree: %s", mismatch)
        print("语料与 agent 时间尺度不一致，拒绝启动：\n  %s" % mismatch, file=sys.stderr)
        return 4

    # The difficulty pick is answered from the config unless the operator says
    # otherwise; "off" refuses it, which parks the game on that screen because
    # the pick has no cancel, so it is only ever the deliberate choice.
    difficulty = args.difficulty if args.difficulty is not None else game.difficulty_preset
    if str(difficulty).strip().lower() in {"", "off", "none"}:
        difficulty = None
        log.info("difficulty pick: refusing it (a round that reaches it will abort)")
    else:
        log.info("difficulty pick: will be answered with %r", difficulty)

    # The action space is the single source of truth for every action-shaped
    # tensor: the policy head width, the IDM head width and the index->mask
    # mapping must all agree on it.  Deriving them here (rather than letting
    # each model default independently) is what keeps three different counts
    # from silently coexisting again.
    probe_env = _make_env(args.backend)
    try:
        action_space = probe_env.action_space
        # Fail fast on a live run rather than letting the loop discover it.  In
        # RPG Maker the ok/cancel keys ARE the keys the policy uses for
        # jump/attack, so the very first thing a live run must not do is press
        # one on a title or menu screen - that is how an unattended run loaded
        # the player's save.
        if args.backend == "cdp":
            reason = probe_env.unsafe_reason()
            if reason and _is_menu_scene(probe_env):
                # An agent that walked into a menu can back out with cancel on
                # its own, so a previous run's accident is not a reason to refuse
                # to start.  This presses cancel ONLY (never ok, never a
                # direction) and only on a scene agent.js lists as a menu.
                escaped = probe_env.escape_menu(max_presses=6)
                log.info("menu escape before start: %s", escaped)
                reason = probe_env.unsafe_reason()
            if reason and not probe_env.is_confirm_scene():
                log.error("refusing to start a live run: %s", reason)
                print(
                    "拒绝启动实机循环：%s\n"
                    "请先在游戏里读档进入可操作的地图，再重试。" % reason,
                    file=sys.stderr,
                )
                return 3
            if reason:
                # An allow-listed confirm screen clears with a single ok, so the
                # round can start here instead of demanding manual intervention.
                log.info("starting from a confirm scene (one ok will clear it): %s", reason)
    finally:
        probe_env.close()
    log.info("action space: %d masks", len(action_space))

    policy = load_or_init_policy(args.policy, _policy_config(args, len(action_space)))
    idm = load_or_init_idm(args.idm, _idm_config(args, len(action_space)))

    from ash.memory.embeddings import FrameEmbedder

    device = str(resolve_device(args.device))
    embedder = FrameEmbedder(device)

    # The corpus is embedded exactly once and the result serves both consumers:
    # K is fit on these embeddings (the paper derives K from the internet corpus
    # D^I), and the very same matrices are the retrieval index step 2 scores
    # against.  Embedding twice would let the two disagree about what a corpus
    # frame looks like, and it doubles the only expensive part of startup.
    #
    # K must exist before the first inference step because runner.run() calls
    # classify() on every 4th frame; the first bootstrap only runs after a whole
    # round of inference has finished, which is too late.
    index = _corpus_embeddings(args.corpus_index, args.corpus, embedder,
                               args.recordings, args.corpus_crops)
    kdm = _key_moment_model(args, embedder, index)

    def runner(**kw):
        envs = [_make_env(args.backend, difficulty) for _ in range(args.num_agents)]
        try:
            r = InferenceRunner(
                lambda: envs[0], policy, kdm,
                action_space=action_space,
                image_size=args.image_size, device=device,
                max_steps=args.max_steps,
                # One action advances the agent's timestep, which is the same
                # interval the corpus is sampled at.  Leaving this at 1 frame
                # made the IDM learn 1/60 s dynamics and apply them to frames
                # 2 s apart.
                frame_skip=game.control_frame_skip,
            )
            return r.run(
                envs,
                delta=kw["delta"],
                timeout_s=kw["timeout_s"],
                random_steps=kw.get("random_steps", 0),
                random_seed=kw.get("random_seed", 0),
            )
        finally:
            for e in envs:
                e.close()

    # Loaded once, sampled per round: the recording is 677 MB and reading it
    # every round would cost more than the training it feeds.
    replay_demo = None
    if args.idm_replay:
        from ash.train.demo_mapping import load_demo

        replay_demo = load_demo(args.idm_replay)
        log.info("idm replay: %d human frames from %s (%d steps per round)",
                 len(replay_demo["observations"]), args.idm_replay,
                 args.idm_replay_steps)

    def bootstrap_fn(**kw):  # full bootstrap; corpus loading lives here
        from ash.loop.bootstrap import BootstrapConfig, Bootstrapper
        from ash.train.demo_mapping import sample_replay

        replay = []
        if replay_demo is not None:
            # The round index is the seed, so successive rounds replay
            # different parts of the recording instead of the same slice.
            replay = sample_replay(replay_demo, action_space,
                                   steps=args.idm_replay_steps,
                                   seed=int(kw.get("round_index", 0)))
            log.info("idm replay: %d transitions from the demonstrations",
                     sum(len(t["act"]) for t in replay))

        b = Bootstrapper(BootstrapConfig(image_size=args.image_size, device=device,
                         max_policy_steps=args.policy_steps))
        return b.run(
            kw["policy"], kw["idm"], kw["kdm"], kw["trajectories"],
            corpus_loader=_corpus_loader(args.corpus, args.recordings,
                                         getattr(args, "corpus_crops", None)),
            out_dir=Path(kw["out_dir"]),
            # D^R from step 2.  Dropping it here made retrieval a no-op: the
            # bootstrap read the whole corpus and reported a plausible
            # per-round result while the retrieved set changed nothing.
            retrieved_ids=kw.get("retrieved_ids"),
            # Random-policy samples for IDM dynamics coverage (paper Alg 4).
            random_trajectories=kw.get("random_trajectories"),
            # The retrieval index already holds every corpus video's DINOv2
            # matrix; without it the bootstrap re-embeds D^R twice per round.
            corpus_embeddings=dict(index),
            # Keeping the human dynamics in the IDM's training set is what stops
            # a round of near-duplicate self-play transitions from overwriting
            # them (measured: val 0.29 -> 3.16 without this).
            replay_trajectories=replay,
        )

    log.info("retrieval index: %d videos", len(index))

    orch = Orchestrator(
        policy, idm, kdm, embedder, index,
        config=LoopConfig(
            delta=args.delta,
            num_agents=args.num_agents,
            max_bootstraps=args.max_bootstraps,
            random_steps=args.random_steps,
            out_dir=out_dir,
        ),
        runner=runner, bootstrap_fn=bootstrap_fn,
    )
    result = orch.run()
    (out_dir / "loop-report.json").write_text(json.dumps(result, indent=2, default=str))
    log.info("done: %d bootstraps", result["bootstraps"])
    if args.backend == "cdp":
        log.info(
            "the game is deliberately left PAUSED (ticker stopped) so an unattended "
            "character is not beaten to death; resume with __ash.pump.resume() in the "
            "page, close(resume=True), or just restart the game"
        )
    return 0


def _is_menu_scene(env) -> bool:
    """True when the env reports a known menu template (agent.js V.MENU_SCENES)."""
    try:
        return bool((env.safety() or {}).get("menu"))
    except Exception:  # noqa: BLE001 - an unreadable scene must abort, not escape
        return False


def _policy_config(args, num_actions: int):
    """Policy config whose head width is the env's action space size.

    `num_actions` is a required argument, not a default: the head width must
    come from the environment, and an optional parameter would let a call site
    silently fall back to DEFAULT_NUM_ACTIONS while the env disagrees.
    """
    from ash.models.ash_policy import AshPolicyConfig

    return AshPolicyConfig(image_size=args.image_size, num_actions=num_actions)


def _idm_config(args, num_actions: int):
    """IDM config whose head width matches the policy's action space."""
    from ash.models.idm import IdmConfig

    return IdmConfig(image_size=args.image_size, num_actions=num_actions)


def _key_moment_model(args, embedder: Any, index: list[tuple[str, np.ndarray]]):
    """K, fit on the internet corpus D^I - reused from a cache when possible.

    Fitting is deterministic given the corpus and the hyperparameters, so a
    cached fit is equivalent to a fresh one (the tests pin that a reloaded
    model answers identically).  It is cached because it is minutes of CPU that
    every run otherwise pays for before the game is even touched: 113 s for the
    six-video corpus after the PCA reduction.

    K must exist before the first inference step because runner.run() calls
    classify() on every 4th frame; the first bootstrap only runs after a whole
    round of inference has finished, which is too late.
    """
    from ash.memory.kdm import KeyMomentModel

    kdm = KeyMomentModel()
    kdm.embedder = embedder
    if not index:
        log.warning(
            "no corpus under %s: key-moment discovery is unfitted, so every "
            "frame counts as noise and the agent will report stuck immediately. "
            "Run scripts/build_corpus.py first for a real run.",
            args.corpus,
        )
        return kdm

    cache = _kdm_cache_path(args)
    meta = _kdm_fingerprint(args, kdm, embedder)
    cached = KeyMomentModel.load(cache, meta)
    if cached is not None:
        cached.embedder = embedder
        log.info("loaded key-moment model from %s (%d kept clusters)",
                 cache, len(cached._kept))
        return cached

    n_frames = sum(len(e) for _, e in index)
    log.info("fitting key-moment model on %d frames from %d corpus videos",
             n_frames, len(index))
    report = kdm.fit(
        np.concatenate([e for _, e in index]),
        [vid for vid, e in index for _ in range(len(e))],
    )
    log.info("key-moment model: %s", report)
    kdm.save(cache, meta)
    log.info("wrote key-moment model to %s", cache)
    return kdm


def _kdm_cache_path(args) -> Path:
    """Sibling of the corpus dir, for the same reason as the embedding cache:
    the loader globs `*.npz` inside the corpus directory, so nothing the run
    produces may be written there."""
    if getattr(args, "kdm_cache", None):
        return Path(args.kdm_cache)
    base = Path(args.corpus or "data/corpus")
    return base.parent / (base.name + "-kdm.pkl")


def _kdm_fingerprint(args, kdm: Any, embedder: Any) -> str:
    """Everything a cached K depends on.

    Includes the clusterer libraries' versions: the pickle stores a fitted
    HDBSCAN, and approximate_predict is the code that runs at every inference
    step, so a version bump must invalidate the file rather than be trusted.
    """
    import sklearn

    return json.dumps(
        {
            "v": 1,
            "corpus": _corpus_fingerprint(args.corpus, embedder.image_size,
                                          args.recordings,
                                          getattr(args, "corpus_crops", None)),
            "params": kdm.hyperparameters(),
            "hdbscan": _dist_version("hdbscan"),
            "sklearn": getattr(sklearn, "__version__", _dist_version("scikit-learn")),
        },
        sort_keys=True,
    )


def _dist_version(name: str) -> str:
    """hdbscan exposes no __version__, so read the installed metadata."""
    import importlib.metadata as md

    try:
        return md.version(name)
    except Exception:  # noqa: BLE001 - a missing version must not break a run
        return "unknown"


def _corpus_embeddings(
    path: str | None,
    corpus_dir: str | None,
    embedder: Any,
    recordings_dir: str | None = None,
    crops_path: str | None = None,
) -> list[tuple[str, np.ndarray]]:
    """One L2-normalized embedding matrix per corpus video.

    The same matrices serve both consumers - K is fit on them and step 2 scores
    against them - so this is built once, not once per consumer.

    Building it costs one DINOv2 pass over the corpus, so the result is cached
    in a sibling file.  The cache deliberately does NOT live inside the corpus
    directory: the loader globs the corpus dir, so a cache written next to the
    videos would be picked up on the next run as a corpus video named "index".
    """
    base = Path(corpus_dir or "data/corpus")
    cache = Path(path) if path else base.parent / (base.name + "-embeddings.npz")
    stamp = _corpus_fingerprint(corpus_dir, embedder.image_size, recordings_dir,
                                crops_path)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            meta = data["__meta__"] if "__meta__" in data.files else None
            if meta is not None and str(meta) == stamp:
                videos = [(k, data[k]) for k in data.files if k != "__meta__"]
                log.info("loaded retrieval index from %s (%d videos)", cache, len(videos))
                return videos
            # A stale index does not fail loudly: retrieval just scores against
            # the wrong frames.  Say why it is being rebuilt instead of silently
            # reusing it.
            log.warning("retrieval index cache %s is stale or unversioned; rebuilding", cache)
    out: list[tuple[str, np.ndarray]] = []
    for vid, frames in _corpus_loader(corpus_dir, recordings_dir, crops_path)():
        if len(frames):
            out.append((vid, embedder.embed(frames)))
    if out:
        np.savez_compressed(cache, __meta__=np.array(stamp), **dict(out))
        log.info("wrote retrieval index to %s (%d videos)", cache, len(out))
    # Hand the embedder's cached blocks back before the round starts.  Embedding
    # the whole corpus is the longest GPU phase of a fresh run, and leaving its
    # allocation cached cost a live round: `live26` rebuilt the index for 16
    # minutes and then died in the bootstrap's first pi update with
    # `CUDA error: out of memory`, while `live25` - same settings, index already
    # cached, so no long embed pass - ran to completion.
    _release_device_cache()
    return out


def _release_device_cache() -> None:
    """Return cached GPU blocks to the driver; no-op without CUDA."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _corpus_dirs(corpus_dir: str | None, recordings_dir: str | None = None) -> list[Path]:
    """The directories that make up D^I, in precedence order.

    Recorded human sessions count as corpus: K is fit on D^I, and a session of the
    rooms the scraped videos never visit is what makes K fire where the agent
    actually is.  Measured before this existed: the agent sat at cosine 0.77-0.92
    to its nearest corpus frame, no corpus frame came within 0.95 of any live
    frame, and `key_moments` was 0 in every round.
    """
    dirs = [Path(p) for p in (corpus_dir or "data/corpus").split(os.pathsep) if p]
    for p in (recordings_dir or "").split(os.pathsep):
        if p and Path(p).is_dir():
            dirs.append(Path(p))
    return dirs


def _corpus_fingerprint(corpus_dir: str | None, image_size: int,
                        recordings_dir: str | None = None,
                        crops_path: str | None = None) -> str:
    """Cheap identity of the corpus + embedding resolution, for the cache.

    Uses file names and sizes rather than loading the frames: the point is to
    notice that the corpus changed, and reading 3 GB of frames to find that out
    would cost more than the embeddings it guards.

    Both extensions are listed.  When the corpus was packed to memory-mappable
    `.npy` and the `.npz` originals were deleted, a fingerprint that only globbed
    `.npz` became empty - so any later change to the corpus would have been
    silently ignored and a stale index reused, which is the exact failure this
    function exists to prevent.
    """
    files = []
    for base in _corpus_dirs(corpus_dir, recordings_dir):
        for path in sorted(base.glob("*.npz")) + sorted(base.glob("*.npy")):
            files.append((str(path), path.stat().st_size))
    files.sort()
    # The crop is part of what the frames MEAN, so it belongs in the identity of
    # the embeddings.  A fingerprint without it would reuse an index built from
    # uncropped frames after the crops changed, silently - the same failure the
    # `.npz`-only glob caused.
    try:
        crops = Path(crops_path).read_text() if crops_path else ""
    except OSError:
        crops = ""
    return json.dumps({"v": 2, "image_size": int(image_size), "files": files,
                       "crops": crops}, sort_keys=True)


def _corpus_loader(corpus_dir: str | None, recordings_dir: str | None = None,
                   crops_path: str | None = None):
    """Yield (video_id, frames) for the corpus, optionally restricted to D^R.

    `ids=None` means the whole corpus D^I (used to fit K and to build the
    retrieval index).  `ids=[...]` means D^R only: the paper refits K and pi on
    the *retrieved* subset, so the bootstrap passes the ids step 2 selected.
    Reading the whole corpus regardless made retrieval a no-op.
    """
    #: Directories searched for observation-only trajectories.  Recorded human
    #: sessions belong here as much as the videos do: K is fit on D^I, and a
    #: session of the rooms the corpus never visits is what makes K fire where the
    #: agent actually is.  Measured before this: the agent sat at cosine 0.77-0.92
    #: to the nearest corpus frame, no corpus frame was within 0.95 of any live
    #: frame, and `key_moments` was 0 in every single round.
    bases = _corpus_dirs(corpus_dir, recordings_dir)
    # Scraped videos carry a splits panel, a banner and black bars; the crop found
    # by `scripts/detect_game_rect.py` removes them and restores the live window's
    # aspect.  Applied on access so the mmap stays an mmap.
    crops = load_crops(crops_path)

    def load(ids: list[str] | None = None):
        want = None if ids is None else set(ids)
        # One stem per file, first directory wins, so a recording deliberately
        # placed in the corpus dir is not shadowed by an empty namesake.
        found: dict[str, Path] = {}
        for base in bases:
            if not base.exists():
                continue
            for path in sorted(base.glob("*.npz")) + sorted(base.glob("*.npy")):
                found.setdefault(path.stem, base)
        for stem, base in sorted(found.items()):
            if want is not None and stem not in want:
                continue
            # Prefer the uncompressed form: memory-mapped frames are clean page
            # cache the kernel can evict, while an npz member is anonymous memory
            # it cannot, and materializing 2.6 GB of it stalled a round in swap.
            packed = base / f"{stem}.npy"
            if packed.exists():
                frames = np.load(packed, mmap_mode="r")
                yield stem, CroppedFrames(frames, crops[stem]) if stem in crops else frames
                continue
            with np.load(base / f"{stem}.npz") as data:
                # `frames` is a scraped video, `observations` a recorded session;
                # both are D^I, they only spell the array differently.
                key = "frames"
                if key not in data.files:
                    key = "observations"
                frames = data[key]
                yield stem, CroppedFrames(frames, crops[stem]) if stem in crops else frames
    return load


class _StopFlag:
    """Set by SIGTERM/SIGINT so the recording is saved instead of discarded.

    A backgrounded recorder cannot be Ctrl-C'd by the person playing, and the
    default SIGTERM action kills the process before `save()` runs - the whole
    session is lost.  Both signals now mean "finish this frame, write the file".
    """

    def __init__(self) -> None:
        self.stop = False
        self._previous: dict[int, Any] = {}

    def install(self) -> None:
        import signal

        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.signal(sig, self._handle)

    def _handle(self, signum, _frame) -> None:
        self.stop = True
        log.warning("signal %d: finishing this frame and saving the recording", signum)

    def requested(self) -> bool:
        return self.stop

    def restore(self) -> None:
        import signal

        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous.clear()


def cmd_record(args: argparse.Namespace) -> int:
    """Record a human playing, without ever dispatching a key.

    This is the data source the IDM actually needs.  Fitting it on the agent's
    own transitions collapsed it to the class prior (val 4.64 against ln(20)=3.00
    from 220 near-duplicate samples); the 24447 frames in `data/idm-human/` are
    what made it work (val 0.29), and this is how more of that gets made.
    """
    import time

    from ash.data.recorder import HumanSessionConfig, HumanSessionRecorder
    from ash.env.cdp_backend import CdpSpeedrunEnv
    from ash.train.demo_mapping import LEGACY_CONTROLS, LEGACY_KEY_CODES

    env = CdpSpeedrunEnv(drive="realtime", resize=(args.size, args.size))
    env.connect(target_index=args.target)
    env.install(seed=args.seed)
    safety = env.safety()
    log.info("scene %s, in gameplay: %s", safety.get("scene"), safety.get("inGameplay"))
    if not safety.get("inGameplay"):
        log.warning("not in Scene_Map: the recording will contain whatever is on screen")

    stop = _StopFlag()
    out = Path(args.out)
    # Samples are spilled here as they arrive.  A background job is killed hard -
    # measured: a 16-minute recording, one file written at the end, nothing left on
    # disk - so the session must not live only in RAM until save().
    session_dir = out.parent / (out.name + ".session")
    cfg = HumanSessionConfig(out=out, fps=args.fps, size=args.size,
                             episode=args.episode, session_dir=session_dir)
    rec = HumanSessionRecorder(env, cfg, LEGACY_CONTROLS, LEGACY_KEY_CODES)
    stop.install()
    rec.install()
    interval = 1.0 / max(args.fps, 1e-6)
    log.info("recording %s at %.1f fps (%d x %d). Play the game now; "
             "Ctrl-C stops and saves.", cfg.out, cfg.fps, cfg.size, cfg.size)
    last = [0.0]

    def tick(n: int, mask: int, frame) -> None:
        now = time.time()
        if now - last[0] >= 5.0:
            last[0] = now
            log.info("recorded %d frames (%.0f s of play)", n, n * interval)

    try:
        frames = rec.run(minutes=args.minutes, on_tick=tick, should_stop=stop.requested)
    finally:
        stop.restore()
        # The player is driving; hand the game back running, unlike the
        # self-play path, which deliberately leaves it paused.
        env.close(resume=True)

    if frames == 0:
        log.error("nothing recorded")
        return 5
    path = rec.save()
    # The npz is the product; the spill is only there so a kill is survivable.
    # Removed only after the archive is on disk and readable.
    try:
        with np.load(path, allow_pickle=True) as check:
            if len(check["control_masks"]) != frames:
                raise ValueError("archive holds %d of %d frames"
                                 % (len(check["control_masks"]), frames))
        shutil.rmtree(session_dir)
    except Exception as exc:       # noqa: BLE001 - never delete data on doubt
        log.warning("kept the spill directory %s (%s)", session_dir, exc)
    masks = np.asarray(rec.masks, dtype=np.int64)
    pressed = int((masks != 0).sum())
    per_key = {
        name: int(sum(1 for m in rec.masks if m & (1 << bit)))
        for bit, name in enumerate(LEGACY_CONTROLS)
    }
    log.info("saved %s: %d frames, %.1f s, %d frames with a key down (%.1f%%)",
             path, frames, frames * interval, pressed, 100.0 * pressed / frames)
    log.info("per-key frame counts: %s", {k: v for k, v in per_key.items() if v})
    log.info("screenshot straddled a key change in %d frames (%.1f%%)",
             rec.alignment_errors, 100.0 * rec.alignment_errors / frames)
    if pressed == 0:
        log.warning("no key was recorded at all - check that the game window had focus")
    log.info("next: uv run python scripts/pack_demos.py %s", path)
    log.info("then: ash pretrain-idm --demos %s --out models/idm-<name>.pt", path)
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    """Recover a spilled recording, e.g. after the process was killed hard."""
    from ash.data.recorder import assemble_session

    path = assemble_session(args.session_dir, args.out)
    with np.load(path, allow_pickle=True) as data:
        n = len(data["control_masks"])
        pressed = int((data["control_masks"] != 0).sum())
    log.info("assembled %d frames (%d with a key down) -> %s", n, pressed, path)
    return 0


def cmd_pretrain_idm(args: argparse.Namespace) -> int:
    """Supervised IDM pretraining on the recorded human demonstrations.

    The ASH bootstrap trains the IDM on the agent's own transitions, and from a
    standing start those transitions are near-duplicates in one small room: the
    value sat at ln(num_classes) - the "predict the prior" solution - for both
    220 and 1060 transitions, and every corpus video came back ~100% one class,
    so pi was never updated at all.

    `vpt-amd-package/data/idm-human.npz` holds 24447 frames of human play with a
    control mask per frame.  That is real (obs[t], obs[t+1], action) data, so the
    IDM can be fitted supervised instead of guessed, which is the only way the
    pseudo-labels stop being a constant.  The masks are translated by
    train/demo_mapping.py (99.1% land exactly on this project's action space).
    """
    from ash.actions.space import ActionSpace
    from ash.loop.bootstrap import BootstrapConfig, Bootstrapper
    from ash.models.idm import IdmConfig, IdmModel, save_idm
    from ash.train.demo_mapping import load_demo, map_legacy_masks

    space = ActionSpace.minimal()
    # `load_demo`, not `np.load`: the same demonstrations exist as an npz and as a
    # packed directory of mmaps, and the other callers already accept both.  Going
    # straight to np.load meant `--demos data/idm-human.npz` failed here once the
    # npz was packed away, while `--idm-replay` with the same path kept working.
    data = load_demo(args.demos)
    obs = data["observations"]
    masks = data["control_masks"]
    episodes = data["episode_ids"]
    # 0 (the default) infers from the recorded frames.  Exposed because the two
    # sources are stored at different native sizes - the 2019-era sessions at 128,
    # a fresh recording at 256 - and whether the extra pixels pay for themselves
    # is a measurement, not an opinion.
    image_size = int(args.image_size or obs.shape[1])
    index, mapping = map_legacy_masks(masks, space)
    log.info("demo mapping: %d frames, %d distinct masks, %d exact, %d snapped, "
             "%d frames whose only presses were dropped keys",
             mapping["frames"], mapping["distinct_legacy_masks"], mapping["exact"],
             mapping["snapped"], mapping["frames_whose_only_presses_were_dropped_keys"])
    if mapping["exact"] < 0.9 * mapping["frames"]:
        log.warning("under 90%% of demo frames map exactly onto the action space")

    trajectories = []
    for episode in np.unique(episodes):
        take = episodes == episode
        frames = obs[take]
        if len(frames) < 3:
            continue
        # The recorded mask is the key state *after* the frame was drawn, which
        # is the convention the legacy IDM trainer was written against, so the
        # label for the pair (t, t+1) is the mask recorded at t+1.
        actions = index[take][1:].astype(np.int64)
        trajectories.append({"obs": frames, "act": actions})
    log.info("demo trajectories: %d episodes, %d transitions",
             len(trajectories), sum(len(t["act"]) for t in trajectories))

    idm = IdmModel(IdmConfig(image_size=image_size, num_actions=len(space)))
    bootstrapper = Bootstrapper(BootstrapConfig(
        image_size=image_size, device=args.device, idm_epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
    ))
    report = bootstrapper.update_idm_from_trajectories(idm, trajectories)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_idm(idm, out)
    log.info("pretrained IDM -> %s (%s)", out, report)
    log.info("use it with: ash run --idm %s ...", out)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.report).read_text())
    print(json.dumps(data, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ash", description=__doc__)
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="environment self-check")
    d.add_argument("--backend", default="fake", choices=["fake", "cdp"])
    d.set_defaults(fn=cmd_doctor)

    r = sub.add_parser("run", help="self-hone loop: infer -> retrieve -> bootstrap")
    r.add_argument("--backend", default="fake", choices=["fake", "cdp"])
    r.add_argument("--policy", default=None, help="existing policy checkpoint")
    r.add_argument("--idm", default=None, help="existing IDM checkpoint")
    r.add_argument("--idm-replay", default=None,
                   help="recorded human demonstrations to replay into every IDM update")
    r.add_argument("--difficulty", default=None,
                   help="answer for the difficulty pick: the option text or a "
                        "1-based number, 'off' to refuse it; default is "
                        "config/game.yaml's difficulty_preset")
    r.add_argument("--policy-steps", type=int, default=300,
                   help="cap on pi optimizer steps per round (0 = unbounded)")
    r.add_argument("--idm-replay-steps", type=int, default=4000,
                   help="demonstration transitions replayed per round")
    r.add_argument("--corpus", default="data/corpus", help="internet video corpus dir")
    r.add_argument("--corpus-crops", default="data/corpus-crops.json",
                   help="crop rects for scraped corpus frames (game viewport only)")
    r.add_argument("--recordings", default="data/recordings",
                   help="dir of recorded human sessions to add to D^I (K coverage)")
    r.add_argument("--corpus-index", default=None, help="precomputed embedding index json")
    r.add_argument("--kdm-cache", default=None,
                   help="fitted key-moment model cache (default: beside the corpus)")
    r.add_argument("--delta", type=int, default=600, help="stuck threshold in steps")
    r.add_argument("--num-agents", type=int, default=1)
    r.add_argument("--image-size", type=int, default=128)
    r.add_argument("--device", default=None)
    r.add_argument("--max-steps", type=int, default=20_000)
    #: The self-hone loop has no natural termination (the paper runs until the
    #: agent clears the game), so a dry run must be able to bound the number of
    #: rounds from the command line.  Without this the default of 64 bootstraps
    #: makes a smoke test indistinguishable from a hang.
    r.add_argument("--max-bootstraps", type=int, default=64)
    #: Random-policy steps per round, added to the IDM's training set (paper
    #: Alg 4 step 2).  0 disables the supplement and reproduces the old,
    #: narrow IDM training set.
    r.add_argument("--random-steps", type=int, default=100)
    r.add_argument("--out", default="runs/ash")
    r.set_defaults(fn=cmd_run)

    asm = sub.add_parser("assemble", help="recover a spilled recording into an npz")
    asm.add_argument("session_dir", help="the .session directory written by `ash record`")
    asm.add_argument("--out", required=True)
    asm.set_defaults(fn=cmd_assemble)

    rc = sub.add_parser("record", help="record a human playing, without sending input")
    rc.add_argument("--out", default="data/human-001.npz")
    rc.add_argument("--fps", type=float, default=10.0)
    rc.add_argument("--size", type=int, default=128)
    rc.add_argument("--minutes", type=float, default=0.0,
                    help="stop after this long (0 = until Ctrl-C)")
    rc.add_argument("--episode", type=int, default=0)
    rc.add_argument("--seed", type=int, default=None)
    rc.add_argument("--target", type=int, default=0)
    rc.set_defaults(fn=cmd_record)

    b = sub.add_parser("pretrain-idm",
                       help="supervised IDM pretraining on recorded human demos")
    b.add_argument("--demos", default="data/idm-human.npz",
                   help="recorded demonstrations with control masks")
    b.add_argument("--out", default="models/idm-demo.pt")
    b.add_argument("--epochs", type=int, default=6)
    b.add_argument("--batch-size", type=int, default=64)
    b.add_argument("--lr", type=float, default=3e-4)
    b.add_argument("--image-size", type=int, default=0,
                   help="0 = use the recorded resolution")
    b.add_argument("--device", default=None)
    b.set_defaults(fn=cmd_pretrain_idm)

    e = sub.add_parser("eval", help="print a loop/eval report")
    e.add_argument("report")
    e.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
